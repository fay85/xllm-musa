// copied from https://github.com/mlc-ai/tokenizers-cpp/blob/v0.1.1/rust/src/lib.rs

// A simple C wrapper of tokenzier library
use std::{collections::HashMap, str::FromStr};
use std::fs;
use std::ffi::{c_char, CStr};
use std::io;
use tokenizers::tokenizer::{Tokenizer, NormalizedString, Normalizer};

pub struct TokenizerWrapper {
    tokenizer: Tokenizer,
    decode_str: String,
    id_to_token_result: String,
}

pub type Vocab = HashMap<String, u32>;
pub type Merges = Vec<(String, String)>;

#[repr(C)]
pub struct TokenizerEncodeResult {
    token_ids: *mut u32,
    len: usize,
}

fn read_file_as_u8(path: &str) -> Result<Vec<u8>, io::Error> {
    fs::read(path)
}

impl TokenizerWrapper {
    pub fn from_str(json: &str) -> TokenizerWrapper {
        let mut tokenizer: Tokenizer = Tokenizer::from_str(json).unwrap().into();
        // `tokenizer.json` may contain export-time truncation/padding policies
        // (for example a fixed `max_length` / `pad_to` value from an offline
        // preprocessing or calibration pipeline). xllm uses this tokenizer for
        // online serving, where prompt length, padding, and batching are
        // controlled by the server/runtime instead of by HuggingFace tokenizer
        // defaults embedded in model artifacts.
        //
        // Clearing these policies avoids silently truncating prompts or padding
        // every request to a fixed length when loading tokenizer.json from such
        // exports.
        //
        // Behavior change / risk:
        // - tokenization here may differ from raw HuggingFace `Tokenizer` behavior
        //   for models whose tokenizer.json intentionally embeds truncation or
        //   padding settings;
        // - workflows that rely on tokenizer-level fixed-length preprocessing
        //   should apply truncation/padding explicitly outside xllm.
        tokenizer.with_padding(None);
        tokenizer.with_truncation(None).unwrap();
        TokenizerWrapper {
            tokenizer,
            decode_str: String::new(),
            id_to_token_result: String::new(),
        }
    }

    pub fn encode(&mut self, text: &str, add_special_tokens: bool) -> Vec<u32> {
        let encoded = self.tokenizer.encode(text, add_special_tokens).unwrap();
        return encoded.get_ids().to_vec();
    }

    pub fn encode_batch(&mut self, texts: Vec<&str>, add_special_tokens: bool) -> Vec<Vec<u32>> {
        let results = self.tokenizer.encode_batch(texts, add_special_tokens).unwrap()
            .into_iter()
            .map(|encoded| encoded.get_ids().to_vec())
            .collect::<Vec<Vec<u32>>>();
        return results;
    }

    pub fn decode(&mut self, ids: &[u32], skip_special_tokens: bool) {
        self.decode_str = self.tokenizer.decode(ids, skip_special_tokens).unwrap();
    }
}

#[no_mangle]
extern "C" fn tokenizers_new_from_str(input_cstr: *const u8, len: usize) -> *mut TokenizerWrapper {
    unsafe {
        let json = &String::from_utf8_lossy(std::slice::from_raw_parts(input_cstr, len));
        return Box::into_raw(Box::new(TokenizerWrapper::from_str(json)));
    }
}

#[no_mangle]
extern "C" fn tokenizers_new_from_path(path: *const c_char) -> *mut TokenizerWrapper {
    let c_str = unsafe { CStr::from_ptr(path) };
    let path_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => panic!("Failed to convert C path string to Rust string"),
    };

    match read_file_as_u8(path_str) {
        Ok(bytes) => {
            return tokenizers_new_from_str(bytes.as_ptr(), bytes.len());
        }
        Err(_) => {
            panic!("Failed to read tokenizer file.");
        }
    }
}

#[no_mangle]
extern "C" fn tokenizers_encode(
    handle: *mut TokenizerWrapper,
    input_cstr: *const u8,
    len: usize,
    add_special_tokens: i32,
    out_result: *mut TokenizerEncodeResult,
) {
    unsafe {
        let input_data = std::str::from_utf8(std::slice::from_raw_parts(input_cstr, len)).unwrap();
        let encoded = (*handle).encode(input_data, add_special_tokens != 0);
        let len = encoded.len();
        *out_result = TokenizerEncodeResult {
            token_ids: Box::into_raw(encoded.into_boxed_slice()) as *mut u32,
            len: len,
        };
    }
}

#[no_mangle]
extern "C" fn tokenizers_encode_batch(
    handle: *mut TokenizerWrapper,
    input_cstr: *const *const u8,
    input_len: *const usize,
    num_seqs: usize,
    add_special_tokens: i32,
    out_result: *mut TokenizerEncodeResult,
) {
    unsafe {
        let input_data = (0..num_seqs)
            .map(|i| {
                std::str::from_utf8(std::slice::from_raw_parts(*input_cstr.offset(i as isize), *input_len.offset(i as isize))).unwrap()
            })
            .collect::<Vec<&str>>();
        let encoded_batch = (*handle).encode_batch(input_data, add_special_tokens != 0);
        for (i, encoded) in encoded_batch.into_iter().enumerate() {
            let len = encoded.len();
            let result = TokenizerEncodeResult {
                token_ids: Box::into_raw(encoded.into_boxed_slice()) as *mut u32,
                len: len,
            };
            *out_result.offset(i as isize) = result;
        }
    }
}

#[no_mangle]
extern "C" fn tokenizers_free_encode_results(results: *mut TokenizerEncodeResult, num_seqs: usize) {
    unsafe {
        let slice = std::slice::from_raw_parts_mut(results, num_seqs);
        for result in &mut *slice {
            drop(Box::from_raw(std::slice::from_raw_parts_mut(result.token_ids, result.len)));
        }
    }
}

#[no_mangle]
extern "C" fn tokenizers_decode(
    handle: *mut TokenizerWrapper,
    input_ids: *const u32,
    len: usize,
    skip_special_tokens: i32,
    out_cstr: *mut *mut u8,
    out_len: *mut usize,
) {
    unsafe {
        let input_data = std::slice::from_raw_parts(input_ids, len);
        (*handle).decode(input_data, skip_special_tokens != 0);

        *out_cstr = (*handle).decode_str.as_mut_ptr();
        *out_len = (&(*handle).decode_str).len();
    }
}

#[no_mangle]
extern "C" fn tokenizers_free(wrapper: *mut TokenizerWrapper) {
    unsafe {
        drop(Box::from_raw(wrapper));
    }
}

#[no_mangle]
extern "C" fn tokenizers_get_vocab_size(handle: *mut TokenizerWrapper, size: *mut usize) {
    unsafe {
        *size = (*handle).tokenizer.get_vocab_size(true);
    }
}

#[no_mangle]
extern "C" fn tokenizers_id_to_token(
    handle: *mut TokenizerWrapper,
    id: u32,
    out_cstr: *mut *mut u8,
    out_len: *mut usize,
) {
    unsafe {
        let str = (*handle).tokenizer.id_to_token(id);
        (*handle).id_to_token_result = match str {
            Some(s) => s,
            None => String::from(""),
        };

        *out_cstr = (*handle).id_to_token_result.as_mut_ptr();
        *out_len = (&(*handle).id_to_token_result).len();
    }
}

#[no_mangle]
extern "C" fn tokenizers_token_to_id(
    handle: *mut TokenizerWrapper,
    token: *const u8,
    len: usize,
    out_id: *mut i32,
) {
    unsafe {
        let token: &str = &String::from_utf8_lossy(std::slice::from_raw_parts(token, len));
        let id = (*handle).tokenizer.token_to_id(token);
        *out_id = match id {
            Some(id) => id as i32,
            None => -1,
        };
    }
}

// Apply the tokenizer's configured normalizer without encoding or adding tokens.
#[no_mangle]
extern "C" fn tokenizers_normalize(
    handle: *mut TokenizerWrapper,
    data: *const u8,
    len: usize,
    out_data: *mut *const u8,
    out_len: *mut usize,
) -> i32 {
    unsafe {
        *out_data = std::ptr::null();
        *out_len = 0;
        let wrapper = &mut *handle;
        wrapper.decode_str.clear();
        let text = match std::str::from_utf8(std::slice::from_raw_parts(data, len)) {
            Ok(text) => text,
            Err(_) => return 1,
        };
        let mut normalized = NormalizedString::from(text);
        if let Some(normalizer) = wrapper.tokenizer.get_normalizer() {
            match normalizer.normalize(&mut normalized) {
                Ok(()) => {}
                Err(_) => return 2,
            }
        }
        wrapper.decode_str = normalized.get().to_owned();
        *out_data = wrapper.decode_str.as_ptr();
        *out_len = wrapper.decode_str.len();
        0
    }
}

#[cfg(test)]
mod normalize_tests {
    use super::*;

    fn tokenizer() -> TokenizerWrapper {
        TokenizerWrapper::from_str(
            r#"{"version":"1.0","truncation":null,"padding":null,
            "added_tokens":[],"normalizer":{"type":"Sequence","normalizers":[
              {"type":"Lowercase"},{"type":"Strip","strip_left":true,"strip_right":true}]},
            "pre_tokenizer":null,"post_processor":null,"decoder":null,
            "model":{"type":"WordLevel","vocab":{"[UNK]":0},"unk_token":"[UNK]"}}"#,
        )
    }

    #[test]
    fn normalizes_without_encoding() {
        let mut wrapper = tokenizer();
        let text = "  ÉCOLE  ";
        let mut data = std::ptr::null();
        let mut len = 0;
        assert_eq!(
            tokenizers_normalize(
                &mut wrapper,
                text.as_ptr(),
                text.len(),
                &mut data,
                &mut len,
            ),
            0
        );
        assert_eq!(
            unsafe { std::slice::from_raw_parts(data, len) },
            "école".as_bytes()
        );
    }

    #[test]
    fn empty_normalized_result_is_success() {
        let mut wrapper = tokenizer();
        let text = "  ";
        let mut data = std::ptr::null();
        let mut len = 99;
        assert_eq!(
            tokenizers_normalize(
                &mut wrapper,
                text.as_ptr(),
                text.len(),
                &mut data,
                &mut len,
            ),
            0
        );
        assert!(!data.is_null());
        assert_eq!(len, 0);
    }

    #[test]
    fn invalid_utf8_clears_previous_output() {
        let mut wrapper = tokenizer();
        let mut data = std::ptr::null();
        let mut len = 0;
        assert_eq!(
            tokenizers_normalize(&mut wrapper, b"OK".as_ptr(), 2, &mut data, &mut len),
            0
        );
        assert_eq!(len, 2);
        let invalid = [0xff];
        assert_eq!(
            tokenizers_normalize(
                &mut wrapper,
                invalid.as_ptr(),
                invalid.len(),
                &mut data,
                &mut len,
            ),
            1
        );
        assert!(data.is_null());
        assert_eq!(len, 0);
        assert!(wrapper.decode_str.is_empty());
    }
}
