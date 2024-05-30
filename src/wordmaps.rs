use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::iter::FromIterator;
use std::ops::Deref;

use ahash::{HashMap, HashMapExt};
use bytecheck::CheckBytes;
use rkyv::{Archive, Deserialize, Serialize};

/// A 32 bit sized pointer to a given word.
///
/// This is used so much we want to reduce the size of our points as much as possible.
/// 32 bits is really all that we require as any larger than a 32 bit length array wont
/// fit in memory or be able to be used regardless.
#[derive(Archive, Deserialize, Serialize, Copy, Clone)]
#[archive(compare(PartialEq))]
#[archive_attr(derive(Debug, CheckBytes))]
pub struct WordRef(u32);

impl Debug for WordRef {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "WordRef(index={})", self.0)
    }
}

/// Is is required because archived types dont implement the methods
/// we define. And for safety we want to hide exposing the inner buffer.
pub trait WordRepr {
    fn as_str(&self) -> &str;
}

impl WordRepr for rkyv::Archived<Word> {
    #[inline]
    fn as_str(&self) -> &str {
        unsafe { std::str::from_utf8_unchecked(self.0.as_ref()) }
    }
}

/// A wrapper around a array of Bytes.
///
/// These are expected to be ascii encoded, although they could potentially be uft-8
/// the rest of the system expects it to be ascii encoded.
///
/// If the string is not ascii encoded this wrapper can become UB as soon as you call
/// as_str which performs an checked transmute.
///
/// The Archived variant of this type only implements `as_str` and derives Debug and EQ.
#[derive(Archive, Deserialize, Serialize, Clone, Default)]
#[archive(compare(PartialEq))]
#[archive_attr(derive(Debug, CheckBytes))]
pub struct Word(Box<[u8]>);

impl WordRepr for Word {
    #[inline]
    fn as_str(&self) -> &str {
        unsafe { std::str::from_utf8_unchecked(self.0.as_ref()) }
    }
}

impl Debug for Word {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Deref for Word {
    type Target = Box<[u8]>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<String> for Word {
    fn from(s: String) -> Self {
        Word(s.into_boxed_str().into_boxed_bytes())
    }
}

impl From<Vec<u8>> for Word {
    fn from(s: Vec<u8>) -> Self {
        Word(s.into_boxed_slice())
    }
}

impl From<&str> for Word {
    fn from(s: &str) -> Self {
        Word(s.to_owned().into_boxed_str().into_boxed_bytes())
    }
}

impl From<&[u8]> for Word {
    fn from(s: &[u8]) -> Self {
        Word(s.to_vec().into_boxed_slice())
    }
}

impl Display for Word {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl PartialEq<Self> for Word {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl PartialEq<[u8]> for Word {
    fn eq(&self, other: &[u8]) -> bool {
        (*self.0).eq(other)
    }
}

impl PartialEq<str> for Word {
    fn eq(&self, other: &str) -> bool {
        (*self.0).eq(other.as_bytes())
    }
}

impl Eq for Word {}

impl Hash for Word {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state)
    }
}

/// A purely memory word map.
///
/// This tries to be very space efficient without sacrificing on access time or
/// read performance.
///
/// This essentially turns a Mapping of `String -> Vec<String>` into a mapping of
/// `u64 (hash of the string) -> Box<[u32]>` and then heavily de-duplicates words which
/// are then inserted as the `word_references` this is just a array containing a `Word` each
/// `WordRef` is just a index to this array in order to retrieve words.
#[derive(Archive, Deserialize, Serialize, Default)]
#[archive_attr(derive(CheckBytes))]
pub struct WordMap {
    data: HashMap<u64, Box<[WordRef]>>,
    word_references: Box<[Word]>,
}

impl Debug for WordMap {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.data)
    }
}

impl WordMap {
    /// Gets the word which is located at the given `WordRef` pointer.
    #[inline]
    pub fn word_at(&self, word_ref: &WordRef) -> &Word {
        unsafe { self.word_references.get_unchecked(word_ref.0 as usize) }
    }

    /// Gets a possible array of references to words that are associated with
    /// the given word.
    #[inline]
    pub fn get(&self, word: &str) -> Option<&[WordRef]> {
        self.data.get(&hash_string(word)).map(|v| v.as_ref())
    }

    /// Creates a new `MemBackedWordMap` from a given dictionary.
    pub fn with_dictionary<K: AsRef<str>>(
        mut dictionary: HashMap<K, Vec<String>>,
    ) -> Self {
        let (ref_words, lookup) = {
            let mut lookup_index: HashMap<String, u32> = HashMap::new();
            let mut ref_words = Vec::new();
            for words in dictionary.values() {
                for word in words {
                    if !lookup_index.contains_key(word) {
                        ref_words.push(Word::from(word.clone()));
                        lookup_index.insert(word.clone(), (ref_words.len() - 1) as u32);
                    }
                }
            }

            let slice = Vec::from_iter(ref_words.into_iter()).into_boxed_slice();

            (slice, lookup_index)
        };

        let mut dict = HashMap::from_iter(dictionary.drain().map(|(k, v)| {
            let v: Vec<_> = v
                .into_iter()
                .map(|w| {
                    let ptr = lookup.get(&w).unwrap();
                    WordRef(*ptr)
                })
                .collect();

            (hash_string(k.as_ref()), v.into_boxed_slice())
        }));

        dict.shrink_to_fit();

        Self {
            data: dict,
            word_references: ref_words,
        }
    }
}

#[inline]
fn hash_string(s: &str) -> u64 {
    let mut hasher = ahash::AHasher::default();
    s.hash(&mut hasher);

    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_words() -> HashMap<String, Vec<String>> {
        HashMap::from_iter(
            vec![
                ("hello".into(), vec!["foo".into(), "bar".into()]),
                ("world".into(), vec!["foo".into(), "baz".into()]),
            ]
            .into_iter(),
        )
    }

    #[test]
    fn test_basic_map() {
        let words = get_words();
        let map = WordMap::with_dictionary(words);

        let word = map.get("hello");
        assert!(word.is_some());

        let word = word.unwrap();
        assert_eq!(word.len(), 2);
    }

    #[test]
    fn bench_basic_map() {
        let words = get_words();
        let map = WordMap::with_dictionary(words);

        let start = std::time::Instant::now();
        for _ in 0..1_000 {
            let word = map.get("hello");
            assert!(word.is_some());

            let word = word.unwrap();
            assert_eq!(word.len(), 2);
        }
        println!("{:?} {:?}/iter", start.elapsed(), start.elapsed() / 1_000);
    }
}
