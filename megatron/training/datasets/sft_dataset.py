# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import atexit
import csv
import math
from collections import Counter
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.megatron_dataset import LowLevelDataset, MegatronDataset
from megatron.core.datasets.utils import Split
from megatron.training.datasets.utils import load_json_arg

IGNORE_INDEX = -100


def _pad_id(tokenizer) -> int:
    """Pad id of Dragon's packs: the tokenizer's pad, else its EOD (Qwen tokenizers have no pad
    under the nemotron prompt formats)."""
    pad = tokenizer.pad
    return pad if pad is not None else tokenizer.eod


class SFTLowLevelDataset:
    """The low-level dataset loading jsonl data for SFT

    Args:
        dataset_path (str): The path to jsonl data
            Each line of the jsonl must have key "messages" (List[Dict]),
            which is a sequence of system/user/assistant messages.
            Must be in the following format:
            [
                {"role": "system", "content": "something"},
                {"role": "user", "content": "something1"},
                {"role": "assistant", "content": "something2"},
            ]
            A jsonl line can contain multiple conversations packed together into on list. Each
            conversation starts with the system role, and conversations can have multiple turns
            of the user and assistant roles.
    """

    def __init__(self, dataset_path: str) -> None:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "SFTDataset currently requires datasets library to be installed"
            )
        self.dataset = load_dataset("json", data_files=dataset_path, split="all")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        # Dragon: the whole record (packed pre-tokenized records have no "messages").
        return self.dataset[idx]


class SFTDataset(MegatronDataset):
    """The dataset used during SFT"""

    def __init__(
        self,
        dataset: LowLevelDataset,
        dataset_path: Optional[str],
        indices: np.ndarray,
        num_samples: Optional[int],
        index_split: Split,
        config: GPTDatasetConfig,
    ) -> None:
        super().__init__(dataset, dataset_path, indices, num_samples, index_split, config)
        # Pre-calculate padding divisor to avoid redundant computation in get_item
        self.padding_divisor = self._calculate_padding_divisor()

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: LowLevelDataset) -> int:
        return len(low_level_dataset)

    @staticmethod
    def build_low_level_dataset(dataset_path: str, config: GPTDatasetConfig) -> LowLevelDataset:
        return SFTLowLevelDataset(dataset_path)

    def __len__(self) -> int:
        return self.num_samples

    def _split_conversations(self, merged_conversations):
        split_conversations = []
        current = []
        for msg in merged_conversations:
            # Whenever we see a new system message, start a new conversation
            if msg["role"] == "system":
                if current:  # If previously accumulating a conversation, then store it
                    split_conversations.append(current)
                current = [msg]  # Then start the new conversation
            else:
                current.append(msg) # Continue accumulating the current conversation
        if current:  # Store any remaining conversation
            split_conversations.append(current)
        return split_conversations

    def _calculate_padding_divisor(self) -> int:
        """
            Calculate the divisor used for sequence padding.
            tp_pad = tp_size * 2 if tp_size > 1 else 1
            cp_pad = cp_size * 2 if cp_size > 1 else 1
            cp_pad = cp_pad * dp_size if hybrid_cp else cp_pad
            divisor = cp_pad * tp_pad
        """
        if self.config.hybrid_context_parallel:
            # Hybrid CP: consider both CP and DP
            cp_pad = self.config.data_parallel_size * self.config.context_parallel_size * 2
        else:
            # Standard CP: only consider CP
            cp_pad = self.config.context_parallel_size * 2 if self.config.context_parallel_size > 1 else 1
        tp_pad = self.config.sequence_parallel_size if self.config.sequence_parallel_size > 0 else 1
        divisor = cp_pad * tp_pad
        # TODO(tailaim): do we need to pad for FP8 execution?
        # divisor = ((divisor + 15) // 16) * 16
        return divisor

    def _getitem_packed(self, record, max_seq_len, L, tokenizer) -> Dict[str, Any]:
        """Build a training sample from a pre-tokenized packed record.

        record = {'input_ids', 'loss_mask', 'cu_seqlens', 'tokens_length'} where
        cu_seqlens are document boundaries WITHIN the pack (in token units of the
        unshifted sequence). We pad to L = max_seq_len + 1, causal-shift, build
        per-document position_ids (reset at each boundary), and pass cu_seqlens +
        max_seqlen through so get_batch can populate packed_seq_params and the
        Mamba/attention kernels reset state per document.
        """
        ids  = list(record['input_ids'])
        mask = list(record['loss_mask'])
        cu   = list(record['cu_seqlens'])
        num_tokens = int(record['tokens_length'])

        # truncate to L (packer already bounds to <= max_seq_len, but be safe)
        if len(ids) > L:
            ids = ids[:L]; mask = mask[:L]
            cu = [c for c in cu if c < L] + [L]
        # pad to L
        pad = L - len(ids)
        if pad > 0:
            ids  = ids  + [_pad_id(tokenizer)] * pad
            mask = mask + [0.0] * pad

        ids  = torch.as_tensor(ids,  dtype=torch.long)
        mask = torch.as_tensor(mask, dtype=torch.float32)
        # causal shift: ids has length L = max_seq_len + 1; tokens/labels/loss_mask
        # are length max_seq_len. cu_seqlens (built on the unshifted ids in [0, L])
        # must therefore be expressed in the SHIFTED model-input coordinate
        # [0, max_seq_len] to match the pretraining contract (where cu_seqlens is
        # computed on the already-shifted `tokens`). We map an unshifted boundary
        # b (start of a document in `ids`) to the shifted index max(0, b-?)... in
        # practice the shift drops index 0, so boundary b in ids maps to b in the
        # labels frame; we clamp the final boundary to max_seq_len and dedup.
        tokens    = ids[:-1].contiguous()
        labels    = ids[1:].contiguous()
        loss_mask = mask[1:].contiguous()

        # Boundaries in the shifted (model-input) coordinate: clamp to max_seq_len
        # and drop any degenerate (zero-length) segments after clamping.
        cu_shift = []
        for c in cu:
            c = min(int(c), max_seq_len)
            if not cu_shift or c != cu_shift[-1]:
                cu_shift.append(c)
        if cu_shift[0] != 0:
            cu_shift = [0] + cu_shift
        if cu_shift[-1] != max_seq_len:
            cu_shift.append(max_seq_len)

        # per-document position_ids: 0..len-1 within each packed doc (shifted frame)
        position_ids = torch.zeros(max_seq_len, dtype=torch.long)
        for a, b in zip(cu_shift[:-1], cu_shift[1:]):
            if b > a:
                position_ids[a:b] = torch.arange(b - a, dtype=torch.long)

        cu_seqlens = torch.as_tensor(cu_shift, dtype=torch.int32)
        seg = (cu_seqlens[1:] - cu_seqlens[:-1])
        max_seqlen = torch.tensor(int(seg.max().item()) if seg.numel() else max_seq_len,
                                  dtype=torch.int32)

        ret = {
            'tokens': tokens,
            'labels': labels,
            'loss_mask': loss_mask,
            'position_ids': position_ids,
            'num_tokens': torch.tensor(num_tokens),
            'cu_seqlens': cu_seqlens,
            'max_seqlen': max_seqlen,
        }
        if self.config.create_attention_mask:
            ret['attention_mask'] = None
        return ret

    def _getitem_dragon_conversation(self, record, max_seq_len, L, tokenizer) -> Dict[str, Any]:
        """Dragon's unpacked path: one conversation {"messages", "tokens_length"[, "chat_template_kwargs"]},
        loss mask from the HF chat template's assistant mask, no cu_seqlens (bucketed in get_batch)."""
        conversation_list = record
        num_tokens = conversation_list['tokens_length']
        # library SFTTokenizer (megatron.core.tokenizers): (input_ids, assistant mask)
        lib_tok = getattr(tokenizer, '_tokenizer', tokenizer)
        tokens_conv, loss_mask = lib_tok.tokenize_sample(conversation_list)
        # truncation
        if len(tokens_conv) > L:
            tokens_conv = tokens_conv[:L]
            loss_mask = loss_mask[:L]
        # padding
        padding_len = L - len(tokens_conv)
        if padding_len:
            tokens_conv = np.array(list(tokens_conv) + [_pad_id(tokenizer)] * padding_len, dtype=np.int64)
            loss_mask   = np.array(list(loss_mask)   + [0.]            * padding_len, dtype=np.float32)
        else:
            tokens_conv = np.array(tokens_conv, dtype=np.int64)
            loss_mask   = np.array(loss_mask, dtype=np.float32)
        # torch conversion
        tokens_conv = torch.as_tensor(tokens_conv, dtype=torch.long)
        loss_mask   = torch.as_tensor(loss_mask,   dtype=torch.float32)
        # shift for causal lm
        tokens       = tokens_conv[:-1].contiguous()
        labels       = tokens_conv[1:].contiguous()
        loss_mask    = loss_mask[1:].contiguous()
        position_ids = torch.arange(max_seq_len, dtype=torch.long)

        if self.config.create_attention_mask:
            ret = {
                'tokens': tokens,
                'labels': labels,
                'attention_mask': None,
                'loss_mask': loss_mask,
                'position_ids': position_ids,
                'num_tokens': torch.tensor(num_tokens),
            }
        else:
            ret = {
                'tokens': tokens,
                'labels': labels,
                'loss_mask': loss_mask,
                'position_ids': position_ids,
                'num_tokens': torch.tensor(num_tokens),
            }

        return ret

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        tokenizer = self.config.tokenizer
        pack_length = self.config.sequence_length

        record = self.dataset[int(self.indices[idx % len(self.indices)])]
        # Dragon: pre-tokenized packs {"input_ids","loss_mask","cu_seqlens","tokens_length"} from
        # the offline packer, and single conversations with "tokens_length".
        if isinstance(record, dict) and 'input_ids' in record:
            return self._getitem_packed(record, pack_length, pack_length + 1, tokenizer)
        if isinstance(record, dict) and 'tokens_length' in record:
            return self._getitem_dragon_conversation(
                record, pack_length, pack_length + 1, tokenizer
            )

        merged_conversations = record["messages"] if isinstance(record, dict) else record
        split_conversations = self._split_conversations(merged_conversations)

        def extend_with_padding(tokens, targets, positions, pad_len):
            tokens.extend([pad] * pad_len)
            targets.extend([pad] * pad_len)
            positions.extend(range(positions[-1]+1, positions[-1]+1+pad_len))

        pack_tokens = []
        pack_targets = []
        pack_positions = []
        cu_seqlens = [0]
        eod = tokenizer.eod
        pad = tokenizer.pad
        # TODO(duncan): Track number of convs dropped and/or truncated and amount of end-padding
        for conversation in split_conversations:

            tokens, targets = tokenizer.tokenize_conversation(
                conversation, return_target=True, add_generation_prompt=False
            )

            tokens_list = tokens.tolist()
            targets_list = targets.tolist()


            pack_tokens.extend(tokens_list)
            pack_targets.extend(targets_list)

            assert not self.config.reset_position_ids
            pack_positions.extend(range(len(tokens_list)))

            pad_granularity = self.padding_divisor
            mod_token_count = len(pack_tokens) % pad_granularity
            if mod_token_count != 0:
                pad_len = pad_granularity - mod_token_count
                extend_with_padding(pack_tokens, pack_targets, pack_positions, pad_len)

            # TODO(duncan): Consider also padding to multiple of number of tokens here. This might
            # be needed for efficiency (and potentially set via command-line argument).

            cu_seqlens.append(len(pack_tokens))

            # Handle any necessary truncation
            if len(pack_tokens) >= pack_length + 1:  # +1 here to account for later alignment
                # Truncate on the right
                max_body = pack_length
                pack_tokens = pack_tokens[:max_body]
                pack_targets = pack_targets[:max_body]
                pack_tokens.append(pad)
                pack_targets.append(pad)
                pack_positions = pack_positions[:pack_length+1]
                # Note len({pack_tokens, pack_targets, pack_positions}) should be pack_length + 1
                cu_seqlens[-1] = len(pack_tokens) - 1
                break

        # Handle any necessary padding
        if len(pack_tokens) < pack_length + 1:  # +1 here to account for later alignment
            pad_len = pack_length + 1 - len(pack_tokens)
            extend_with_padding(pack_tokens, pack_targets, pack_positions, pad_len)
            # Note len({pack_tokens, pack_targets, pack_positions}) should be pack_length + 1
            cu_seqlens[-1] = len(pack_tokens) - 1

        assert len(pack_tokens) == pack_length + 1
        assert len(pack_targets) == pack_length + 1
        assert len(pack_positions) == pack_length + 1

        # Align and convert to tensors
        input_ids    = torch.tensor(pack_tokens[:-1],  dtype=torch.int64)
        labels       = torch.tensor(pack_targets[1:], dtype=torch.int64)
        position_ids = torch.tensor(pack_positions[:-1], dtype=torch.int64)

        # Loss mask.
        loss_mask = torch.ones(pack_length, dtype=torch.float32)
        loss_mask[labels == pad] = 0.0  # Mask paddings
        loss_mask[labels == IGNORE_INDEX] = 0.0  # mask prompts

        # TODO(duncan): Optionally create an attention mask
        assert not self.config.create_attention_mask and not self.config.reset_attention_mask
        # attention_mask = None

        assert len(cu_seqlens) >= 2
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32)
        # Calculating max_seqlen here, rather than incrementally above, because of possible
        # effects of truncation and padding
        adjacent_diffs = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = adjacent_diffs.max()  # max_seqlen is a 0-D tensor

        # Pad cu_seqlens to a fixed length so that default_collate can
        # stack samples with different numbers of documents.  Trailing
        # entries are filled with pack_length; the merge helper strips
        # them later.
        padded_cu_seqlens = torch.full(
            (pack_length + 1,), pack_length, dtype=torch.int32,
        )
        padded_cu_seqlens[:cu_seqlens.numel()] = cu_seqlens

        return {
            'tokens': input_ids,
            'labels': labels,
            # 'attention_mask': attention_mask,  # PyTorch collate cannot handle NoneType
            'loss_mask': loss_mask,
            'position_ids': position_ids,
            'cu_seqlens': padded_cu_seqlens,
            'max_seqlen': max_seqlen,
        }


class MockSFTLowLevelDataset:
    """The low-level mock dataset for SFT

    Args:
        mode (str): Either 'file' or 'distribution'.
        **kwargs: Additional arguments depending on mode.
            For mode='file': path (str) - path to a CSV file with sequence lengths.
            For mode='distribution': type (str), min_seq_len (int), max_seq_len (int),
                mean_seq_len (int), and distribution-specific params (e.g. lognormal_sigma).
    """

    seed: int = 0
    """The hard-coded random seed to use to set the NumPy RNG"""

    size: int = 1000000
    """The hard-coded number of sequence to generate"""

    def __init__(self, mode: str, **kwargs) -> None:
        np.random.seed(self.seed)

        if mode == "file":
            # One sequence length per CSV line; non-numeric cells (e.g. a
            # header row) are skipped. Stdlib csv keeps pandas out of the
            # package's import-time dependencies.
            lengths = []
            with open(kwargs["path"], newline="") as f:
                for row in csv.reader(f):
                    for cell in row:
                        cell = cell.strip()
                        if not cell:
                            continue
                        try:
                            lengths.append(int(float(cell)))
                        except ValueError:
                            continue
            self.sequence_lengths = np.array(lengths)
            self.size = len(self.sequence_lengths)
        elif mode == "distribution":
            min_seq_len = kwargs["min_seq_len"]
            max_seq_len = kwargs["max_seq_len"]
            mean_seq_len = kwargs["mean_seq_len"]
            if kwargs["type"] == "lognormal":
                lognormal_sigma = kwargs["lognormal_sigma"]
                self.sequence_lengths = self.generate_lognormal_samples(
                    self.size, mean_seq_len, lognormal_sigma, min_seq_len, max_seq_len
                )
            else:
                raise ValueError(f"Unsupported distribution type {kwargs['type']}")
        else:
            raise ValueError(f"Unsupported mode '{mode}', must be 'file' or 'distribution'")

    def generate_lognormal_samples(self, size, mean, sigma, min_seq_len, max_seq_len):
        mu = np.log(mean) - sigma**2 / 2
        samples = np.random.lognormal(mu, sigma, size)
        samples = np.clip(samples, min_seq_len, max_seq_len)
        return samples.astype(int)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> List[np.ndarray]:
        # the length of sample is 'length', but only length-1 elements are generated here,
        # because an eod token will be appended at the end later in SFTDataset

        length = self.sequence_lengths[idx % self.size]
        sample = np.arange(1, length, dtype=np.int64)
        return sample


class MockSFTDataset(SFTDataset):
    """The mock dataset used during SFT"""

    def __init__(
        self,
        dataset: LowLevelDataset,
        dataset_path: Optional[str],
        indices: np.ndarray,
        num_samples: Optional[int],
        index_split: Split,
        config: GPTDatasetConfig,
    ) -> None:
        super().__init__(dataset, dataset_path, indices, num_samples, index_split, config)

    @staticmethod
    def build_low_level_dataset(dataset_path: str, config: GPTDatasetConfig) -> LowLevelDataset:
        if config.sft_mock_dataset_config_json is None:
            mock_config = {
                "mode": "distribution",
                "type": "lognormal",
                "min_seq_len": config.sequence_length // 2,
                "max_seq_len": config.sequence_length,
                "mean_seq_len": config.sequence_length // 4 * 3,
                "lognormal_sigma": 1.1,
            }
        else:
            mock_config = load_json_arg(config.sft_mock_dataset_config_json)
        return MockSFTLowLevelDataset(**mock_config)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        tokenizer = self.config.tokenizer
        pack_length = self.config.sequence_length
        eod = tokenizer.eod
        pad = tokenizer.pad

        tokens = self.dataset[int(self.indices[idx % len(self.indices)])]

        def extend_with_padding(tokens, targets, positions, pad_len):
            tokens.extend([pad] * pad_len)
            targets.extend([pad] * pad_len)
            positions.extend(range(positions[-1] + 1, positions[-1] + 1 + pad_len))

        # Convert tokens to list and add EOD
        tokens_list = tokens.tolist()
        if tokens_list[-1] != eod:
            tokens_list.append(eod)
        targets_list = list(tokens_list)

        pack_tokens = list(tokens_list)
        pack_targets = list(targets_list)
        pack_positions = list(range(len(tokens_list)))
        cu_seqlens = [0]

        # Pad to padding_divisor alignment
        if self.padding_divisor > 1:
            mod_token_count = len(pack_tokens) % self.padding_divisor
            if mod_token_count != 0:
                pad_len = self.padding_divisor - mod_token_count
                extend_with_padding(pack_tokens, pack_targets, pack_positions, pad_len)

        # Record padded boundary after padding
        cu_seqlens.append(len(pack_tokens))

        # Handle any necessary truncation
        if len(pack_tokens) >= pack_length + 1:  # +1 here to account for later alignment
            max_body = pack_length - 1
            pack_tokens = pack_tokens[:max_body]
            pack_targets = pack_targets[:max_body]
            pack_tokens.extend([eod, pad])
            pack_targets.extend([eod, pad])
            pack_positions = pack_positions[:pack_length + 1]
            cu_seqlens[-1] = len(pack_tokens) - 1

        # Handle any necessary padding
        if len(pack_tokens) < pack_length + 1:  # +1 here to account for later alignment
            pad_len = pack_length + 1 - len(pack_tokens)
            extend_with_padding(pack_tokens, pack_targets, pack_positions, pad_len)
            cu_seqlens[-1] = len(pack_tokens) - 1

        assert len(pack_tokens) == pack_length + 1
        assert len(pack_targets) == pack_length + 1
        assert len(pack_positions) == pack_length + 1

        # Align and convert to tensors
        input_ids = torch.tensor(pack_tokens[:-1], dtype=torch.int64)
        labels = torch.tensor(pack_targets[1:], dtype=torch.int64)
        position_ids = torch.tensor(pack_positions[:-1], dtype=torch.int64)

        # Loss mask
        loss_mask = torch.ones(pack_length, dtype=torch.float32)
        loss_mask[labels == pad] = 0.0
        loss_mask[labels == IGNORE_INDEX] = 0.0

        assert len(cu_seqlens) >= 2
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32)
        # Calculating max_seqlen here because of possible effects of truncation and padding
        adjacent_diffs = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = adjacent_diffs.max()  # max_seqlen is a 0-D tensor

        return {
            'tokens': input_ids,
            'labels': labels,
            'loss_mask': loss_mask,
            'position_ids': position_ids,
            'cu_seqlens': cu_seqlens,
            'max_seqlen': max_seqlen,
        }
