from ..base import BaseMapper, TransformElementType

import unicodedata
from typing import Optional, Any, List
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class TokenizerMapper(BaseMapper):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizerBase,
            input_field: str,
            output_prefix: Optional[str] = None,
            add_special_tokens: Optional[bool] = True,
            max_length: Optional[int] = None,
            is_split_into_words: Optional[bool] = False,
            return_token_type_ids: Optional[bool] = False,
            return_attention_mask: Optional[bool] = True,
            return_overflowing_tokens: Optional[bool] = False,
            return_special_tokens_mask: Optional[bool] = False,
            return_offsets_mapping: Optional[bool] = False,
            return_length: Optional[bool] = False,
            return_word_ids: Optional[bool] = False,
            return_words: Optional[bool] = False,
            **tk_kwargs: Any
    ) -> None:
        super().__init__()

        self.batched = False
        self.tokenizer = tokenizer
        self.prefix = f'{output_prefix}_' if output_prefix else ''

        tk_kwargs = {'add_special_tokens': add_special_tokens,
                     'max_length': max_length,
                     'is_split_into_words': is_split_into_words,
                     **(tk_kwargs or {})}

        input_fields = [input_field]
        output_fields = [f'{self.prefix}input_ids']

        # various options for the tokenizer affect which fields are returned
        tk_kwargs['return_attention_mask'] = return_attention_mask
        if return_attention_mask:
            output_fields.append(f'{self.prefix}attention_mask')

        tk_kwargs['return_token_type_ids'] = return_token_type_ids
        if return_token_type_ids:
            output_fields.append(f'{self.prefix}token_type_ids')

        tk_kwargs['return_overflowing_tokens'] = return_overflowing_tokens
        if return_overflowing_tokens:
            output_fields.append(f'{self.prefix}overflow_to_sample_mapping')

        tk_kwargs['return_special_tokens_mask'] = return_special_tokens_mask
        if return_special_tokens_mask:
            output_fields.append(f'{self.prefix}special_tokens_mask')

        tk_kwargs['return_offsets_mapping'] = return_offsets_mapping
        if return_offsets_mapping:
            output_fields.append(f'{self.prefix}offset_mapping')

        tk_kwargs['return_length'] = return_length
        if return_length:
            output_fields.append(f'{self.prefix}length')

        if 'is_split_into_words' in tk_kwargs and return_word_ids:
            output_fields.append(f'{self.prefix}word_ids')
            if return_words:
                output_fields.append(f'{self.prefix}words')

        self.tokenize_kwargs = tk_kwargs
        self.input_fields = input_fields
        self.output_fields = output_fields

    def transform(self, data: TransformElementType) -> TransformElementType:
        batch_encoding = self.tokenizer(data[self.input_fields[0]], **self.tokenize_kwargs)
        # token_to_word mappings are unfortunately not natively returned by HF.tokenizer
        # so we need to operate separately on the `batch_encoding` object to get this info.
        if f'{self.prefix}word_ids' in self.output_fields:
            word_idss = [
                batch_encoding[sequence_id].word_ids
                for sequence_id in range(len(batch_encoding['input_ids']))
            ]
            if f'{self.prefix}words' in self.output_fields:
                wordss = [
                    [
                        None if word_id is None else data[self.input_fields[0]][word_id]
                        for word_id in word_ids
                    ]
                    for word_ids in word_idss
                ]
            else:
                wordss = None
        else:
            word_idss = None
            wordss = None
            
        # cast to dict
        batch_encoding_as_dict = {f'{self.prefix}{k}': v for k, v in batch_encoding.items()}
        if word_idss is not None:
            batch_encoding_as_dict[f'{self.prefix}word_ids'] = word_idss
            if wordss is not None:
                batch_encoding_as_dict[f'{self.prefix}words'] = wordss
        return dict(batch_encoding_as_dict)


class ValidUnicodeMapper(BaseMapper):
    """Given input_fields of type List[str], replaces invalid Unicode
    characters with something else"""

    def __init__(
            self,
            input_fields: List[str],
            unicode_categories: List[str],
            replace_token: str,
            output_prefix: Optional[str] = None,
    ):
        self.batched = False
        self.input_fields = input_fields
        self.prefix = f'{output_prefix}_' if output_prefix else ''
        self.output_fields = [
            f'{self.prefix}{input_field}' for input_field in self.input_fields
        ]
        self.unicode_categories = unicode_categories
        self.replace_token = replace_token

    def transform(self, data: TransformElementType) -> TransformElementType:
        def _transform(tokens: List[str]) -> List[str]:
            return [
                self.replace_token if all(
                    unicodedata.category(ch) in self.unicode_categories
                    for ch in token
                ) else token
                for token in tokens
            ]
        new_data = {f'{self.prefix}{k}': v if k not in self.input_fields
                    else _transform(v) for k, v in data.items()}
        return new_data
