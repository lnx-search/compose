use std::borrow::Cow;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::{cmp, i64};

use bytecheck::CheckBytes;
use deunicode::deunicode;
use hashbrown::{HashMap, HashSet};
use rkyv::{Archive, Deserialize, Serialize};

use crate::composition::Composition;
use crate::edit_distance;
use crate::suggestion::Suggestion;
use crate::wordmaps::{WordMap, WordRepr};

#[derive(Eq, PartialEq, Debug)]
pub enum Verbosity {
    Top,
    Closest,
    All,
}

mod ascii {
    #[inline]
    pub fn prepare(s: &str) -> String {
        let mut new_word = String::with_capacity(s.len());
        for char in s.chars() {
            if let Some(decoded) = deunicode::deunicode_char(char) {
                new_word.push_str(decoded);

                if decoded.len() > 1 {
                    new_word.push(' ');
                }
            };
        }

        new_word
    }

    #[inline]
    pub fn remove(s: &str, index: usize) -> String {
        let mut x = s.to_string();
        x.remove(index);
        x
    }

    #[inline]
    pub fn slice(s: &str, start: usize, end: usize) -> &str {
        unsafe { s.get_unchecked(start..end) }
    }

    #[inline]
    pub fn suffix(s: &str, start: usize) -> &str {
        slice(s, start, s.len())
    }

    #[inline]
    pub fn at(s: &str, i: isize) -> Option<char> {
        let index = usize::try_from(i).ok()?;
        s.as_bytes().get(index).copied().map(char::from)
    }
}

const WORD_COUNT: i64 = 1_024_908_267_229;
const PREFIX_LENGTH: i64 = 7;

#[derive(Archive, Deserialize, Serialize)]
#[archive_attr(derive(CheckBytes))]
pub struct SymSpell {
    /// Maximum edit distance for doing lookups.
    max_dictionary_edit_distance: i64,
    /// The minimum frequency count for dictionary words to be considered correct spellings.
    count_threshold: i64,
    max_length: usize,
    words: HashMap<String, i64>,
    pub deletes: WordMap,
}

impl Default for SymSpell {
    fn default() -> Self {
        Self {
            max_dictionary_edit_distance: 2,
            count_threshold: 1,
            max_length: 0,
            words: Default::default(),
            deletes: Default::default(),
        }
    }
}

impl SymSpell {
    /// Load multiple dictionary entries from a file of word/frequency count pairs.
    ///
    /// # Arguments
    ///
    /// * `corpus` - The path+filename of the file.
    /// * `term_index` - The column position of the word.
    /// * `count_index` - The column position of the frequency count.
    /// * `separator` - Separator between word and frequency
    pub fn using_dictionary_file(
        &mut self,
        corpus: &str,
        term_index: i64,
        count_index: i64,
        separator: &str,
    ) -> bool {
        if !Path::new(corpus).exists() {
            return false;
        }

        let file = File::open(corpus).expect("file not found");
        let sr = BufReader::new(file);

        let mut frequencies = Vec::new();
        for line in sr.lines() {
            let l = line.unwrap();
            let line_parts: Vec<&str> = l.split(separator).collect();
            let key = &line_parts[term_index as usize];
            let count = line_parts[count_index as usize].parse::<i64>().unwrap();

            frequencies.push((deunicode(&key), count))
        }

        // Safe because we normalize all keys to ascii encoding before passing.
        unsafe { self.using_dictionary_frequencies(frequencies, true) };

        true
    }

    /// Sets the symspell system to use a given set of frequencies.
    ///
    /// # Safety
    /// This is marked as unsafe because it assumed the tokens are already normalized to a purely
    /// ascii standard.
    ///
    /// If you do not ensure that the tokens are ascii this systems becomes UB when trying
    /// to perform lookups.
    pub unsafe fn using_dictionary_frequencies(
        &mut self,
        frequencies: Vec<(String, i64)>,
        compute_non_prefix: bool,
    ) {
        let mut deletes = HashMap::new();
        let mut long_words = HashSet::new();
        let edit_distance = self.max_dictionary_edit_distance;

        self.words = HashMap::new();

        let mut to_compute = Vec::new();
        for (term, count) in frequencies {
            if compute_non_prefix && (term.len() > PREFIX_LENGTH as usize) {
                long_words.insert(term.clone());
            }

            if count < self.count_threshold {
                continue;
            }

            match self.words.get(&term) {
                Some(i) => {
                    let updated_count = if i64::MAX - i > count {
                        i + count
                    } else {
                        i64::MAX
                    };
                    self.words.insert(term.to_string(), updated_count);
                    continue;
                },
                None => {
                    self.words.insert(term.to_string(), count);
                    to_compute.push(term.to_string());
                },
            }

            let key_len = term.len();
            if key_len > self.max_length {
                self.max_length = key_len;
            }
        }

        for word in to_compute {
            let computed_edits = edits_prefix(edit_distance, &word);
            for delete in computed_edits {
                deletes
                    .entry(delete.to_string())
                    .and_modify(|e: &mut Vec<String>| {
                        if !e.contains(&word) {
                            e.push(word.clone());
                        }
                    })
                    .or_insert_with(|| vec![word.clone()]);
            }
        }

        self.deletes = WordMap::with_dictionary(deletes);
    }

    /// Find suggested spellings for a given input word, using the maximum
    /// edit distance specified during construction of the SymSpell dictionary.
    ///
    /// # Arguments
    ///
    /// * `input` - The word being spell checked.
    /// * `verbosity` - The value controlling the quantity/closeness of the retuned suggestions.
    /// * `max_edit_distance` - The maximum edit distance between input and suggested words.
    ///
    /// # Examples
    ///
    /// ```
    /// use compose::{SymSpell, Verbosity};
    ///
    /// let mut symspell = SymSpell::default();
    /// symspell.using_dictionary_file("data/frequency_dictionary_en_82_765.txt", 0, 1, " ");
    /// symspell.lookup("whatver", Verbosity::Top, 2);
    /// ```
    pub fn lookup(
        &self,
        input: &str,
        verbosity: Verbosity,
        max_edit_distance: i64,
    ) -> Vec<Suggestion> {
        if max_edit_distance > self.max_dictionary_edit_distance {
            panic!("max_edit_distance is bigger than max_dictionary_edit_distance");
        }

        let mut suggestions: Vec<Suggestion> = Vec::with_capacity(4);

        let prep_input = ascii::prepare(input);
        let input = prep_input.as_str();
        let input_len = input.len() as i64;

        if (input_len - self.max_dictionary_edit_distance) > self.max_length as i64 {
            return suggestions;
        }

        let mut hashset1: HashSet<Cow<str>> = HashSet::with_capacity(16);
        let mut hashset2: HashSet<Cow<str>> = HashSet::with_capacity(16);

        if self.words.contains_key(input) {
            let suggestion_count = self.words[input];
            suggestions.push(Suggestion::new(input, 0, suggestion_count));

            if verbosity != Verbosity::All {
                return suggestions;
            }
        }

        hashset2.insert(Cow::Borrowed(input));

        let mut max_edit_distance2 = max_edit_distance;
        let mut candidate_pointer = 0;
        let mut candidates: Vec<Cow<str>> = Vec::new();

        let mut input_prefix_len = input_len;

        if input_prefix_len > PREFIX_LENGTH {
            input_prefix_len = PREFIX_LENGTH;
            candidates.push(Cow::Borrowed(ascii::slice(
                input,
                0,
                input_prefix_len as usize,
            )));
        } else {
            candidates.push(Cow::Borrowed(input));
        }

        while candidate_pointer < candidates.len() {
            let candidate = candidates.get(candidate_pointer).unwrap().clone();
            candidate_pointer += 1;
            let candidate_len = candidate.len() as i64;
            let length_diff = input_prefix_len - candidate_len;

            if length_diff > max_edit_distance2 {
                if verbosity == Verbosity::All {
                    continue;
                }
                break;
            }

            if let Some(dict_suggestions) = self.deletes.get(&candidate) {
                for ref_ in dict_suggestions {
                    let suggestion = self.deletes.word_at(ref_);
                    let suggestion_len = suggestion.len() as i64;

                    if suggestion == input {
                        continue;
                    }

                    if (suggestion_len - input_len).abs() > max_edit_distance2
                        || suggestion_len < candidate_len
                        || (suggestion_len == candidate_len
                            && suggestion.as_str() != candidate)
                    {
                        continue;
                    }

                    let sugg_prefix_len = cmp::min(suggestion_len, PREFIX_LENGTH);

                    if sugg_prefix_len > input_prefix_len
                        && sugg_prefix_len - candidate_len > max_edit_distance2
                    {
                        continue;
                    }

                    let distance;

                    if candidate_len == 0 {
                        distance = cmp::max(input_len, suggestion_len);

                        if distance > max_edit_distance2
                            || hashset2.contains(suggestion.as_str())
                        {
                            continue;
                        }
                        hashset2.insert(Cow::Borrowed(suggestion.as_str()));
                    } else if suggestion_len == 1 {
                        distance = if !input.contains(ascii::slice(
                            suggestion.as_str(),
                            0,
                            1,
                        )) {
                            input_len
                        } else {
                            input_len - 1
                        };

                        if distance > max_edit_distance2
                            || hashset2.contains(suggestion.as_str())
                        {
                            continue;
                        }

                        hashset2.insert(Cow::Borrowed(suggestion.as_str()));
                    } else if self.has_different_suffix(
                        max_edit_distance,
                        input,
                        input_len,
                        candidate_len,
                        suggestion.as_str(),
                        suggestion_len,
                    ) {
                        continue;
                    } else {
                        if verbosity != Verbosity::All
                            && !self.delete_in_suggestion_prefix(
                                &candidate,
                                candidate_len,
                                suggestion.as_str(),
                                suggestion_len,
                            )
                        {
                            continue;
                        }

                        if hashset2.contains(suggestion.as_str()) {
                            continue;
                        }
                        hashset2.insert(Cow::Borrowed(suggestion.as_str()));

                        if let Some(d) = edit_distance::compare(
                            input,
                            suggestion.as_str(),
                            max_edit_distance2,
                        ) {
                            distance = d;
                        } else {
                            continue;
                        }
                    }

                    if distance <= max_edit_distance2 {
                        let suggestion_count = match self.words.get(suggestion.as_str())
                        {
                            None => unreachable!(),
                            Some(v) => *v,
                        };
                        let si = Suggestion::new(
                            suggestion.as_str(),
                            distance,
                            suggestion_count,
                        );

                        if !suggestions.is_empty() {
                            match verbosity {
                                Verbosity::Closest => {
                                    if distance < max_edit_distance2 {
                                        suggestions.clear();
                                    }
                                },
                                Verbosity::Top => {
                                    if distance < max_edit_distance2
                                        || suggestion_count > suggestions[0].count
                                    {
                                        max_edit_distance2 = distance;
                                        suggestions[0] = si;
                                    }
                                    continue;
                                },
                                _ => (),
                            }
                        }

                        if verbosity != Verbosity::All {
                            max_edit_distance2 = distance;
                        }

                        suggestions.push(si);
                    }
                }
            }

            if length_diff < max_edit_distance && candidate_len <= PREFIX_LENGTH {
                if verbosity != Verbosity::All && length_diff >= max_edit_distance2 {
                    continue;
                }

                for i in 0..candidate_len {
                    let delete: Cow<str> =
                        Cow::Owned(ascii::remove(&candidate, i as usize));

                    if !hashset1.contains(delete.as_ref()) {
                        hashset1.insert(delete.clone());
                        candidates.push(delete);
                    }
                }
            }
        }

        if suggestions.len() > 1 {
            suggestions.sort();
        }

        suggestions
    }

    /// Find suggested spellings for a given input sentence, using the maximum
    /// edit distance specified during construction of the SymSpell dictionary.
    ///
    /// # Arguments
    ///
    /// * `input` - The sentence being spell checked.
    /// * `max_edit_distance` - The maximum edit distance between input and suggested words.
    ///
    /// # Examples
    ///
    /// ```
    /// use compose::SymSpell;
    ///
    /// let mut symspell = SymSpell::default();
    /// symspell.using_dictionary_file("data/frequency_dictionary_en_82_765.txt", 0, 1, " ");
    /// symspell.lookup_compound("whereis th elove", 2);
    /// ```
    pub fn lookup_compound(&self, input: &str, edit_distance_max: i64) -> String {
        // parse input string into single terms
        let term_list1 = parse_words(&ascii::prepare(input));

        // let mut suggestions_previous_term: Vec<Suggestion> = Vec::new();
        // suggestions for a single term
        let mut suggestions: Vec<Suggestion>;
        let mut suggestion_parts: Vec<Suggestion> = Vec::new();

        // translate every term to its best suggestion, otherwise it remains unchanged
        let mut last_combi = false;

        for (i, term) in term_list1.iter().enumerate() {
            suggestions = self.lookup(term, Verbosity::Top, edit_distance_max);

            //combi check, always before split
            if i > 0 && !last_combi {
                let mut suggestions_combi: Vec<Suggestion> = self.lookup(
                    &format!("{}{}", term_list1[i - 1], term_list1[i]),
                    Verbosity::Top,
                    edit_distance_max,
                );

                if !suggestions_combi.is_empty() {
                    let best1 = suggestion_parts[suggestion_parts.len() - 1].clone();
                    let best2 = if !suggestions.is_empty() {
                        suggestions[0].clone()
                    } else {
                        Suggestion::new(
                            term_list1[1].as_str(),
                            edit_distance_max + 1,
                            10 / (10i64).pow(term_list1[i].len() as u32),
                        )
                    };

                    // if (suggestions_combi[0].distance + 1 < DamerauLevenshteinDistance(term_list1[i - 1] + " " + term_list1[i], best1.term + " " + best2.term))
                    let distance1 = best1.distance + best2.distance;

                    if (distance1 >= 0)
                        && (suggestions_combi[0].distance + 1 < distance1
                            || (suggestions_combi[0].distance + 1 == distance1
                                && (suggestions_combi[0].count
                                    > best1.count / WORD_COUNT * best2.count)))
                    {
                        suggestions_combi[0].distance += 1;
                        let last_i = suggestion_parts.len() - 1;
                        suggestion_parts[last_i] = suggestions_combi[0].clone();
                        last_combi = true;
                        continue;
                    }
                }
            }
            last_combi = false;

            // always split terms without suggestion / never split terms with suggestion ed=0 / never split single char terms
            if !suggestions.is_empty()
                && ((suggestions[0].distance == 0) || (term_list1[i].len() == 1))
            {
                //choose best suggestion
                suggestion_parts.push(suggestions[0].clone());
            } else {
                let mut suggestion_split_best = if !suggestions.is_empty() {
                    //add original term
                    suggestions[0].clone()
                } else {
                    //if no perfect suggestion, split word into pairs
                    Suggestion::empty()
                };

                let term_length = term_list1[i].len();

                if term_length > 1 {
                    for j in 1..term_length {
                        let part1 = ascii::slice(&term_list1[i], 0, j);
                        let part2 = ascii::slice(&term_list1[i], j, term_length);

                        let mut suggestion_split = Suggestion::empty();

                        let suggestions1 =
                            self.lookup(part1, Verbosity::Top, edit_distance_max);

                        if !suggestions1.is_empty() {
                            let suggestions2 =
                                self.lookup(part2, Verbosity::Top, edit_distance_max);

                            if !suggestions2.is_empty() {
                                // select best suggestion for split pair
                                let term = format!(
                                    "{} {}",
                                    suggestions1[0].term, suggestions2[0].term
                                );

                                let distance2 = edit_distance::compare(
                                    &term_list1[i],
                                    &term,
                                    edit_distance_max,
                                )
                                .unwrap_or_else(|| edit_distance_max + 1);
                                suggestion_split.term = term;

                                if !suggestion_split_best.term.is_empty() {
                                    if distance2 > suggestion_split_best.distance {
                                        continue;
                                    }
                                    if distance2 < suggestion_split_best.distance {
                                        suggestion_split_best = Suggestion::empty();
                                    }
                                }

                                // The Naive Bayes probability of
                                // the word combination is the
                                // product of the two word
                                // probabilities: P(AB)=P(A)*P(B)
                                // use it to estimate the frequency
                                // count of the combination, which
                                // then is used to rank/select the
                                // best splitting variant
                                let count2: i64 = ((suggestions1[0].count as f64)
                                    / (WORD_COUNT as f64)
                                    * (suggestions2[0].count as f64))
                                    as i64;

                                suggestion_split.distance = distance2;
                                suggestion_split.count = count2;

                                //early termination of split
                                if suggestion_split_best.term.is_empty()
                                    || suggestion_split.count
                                        > suggestion_split_best.count
                                {
                                    suggestion_split_best = suggestion_split.clone();
                                }
                            }
                        }
                    }

                    if !suggestion_split_best.term.is_empty() {
                        // select best suggestion for split pair
                        suggestion_parts.push(suggestion_split_best.clone());
                    } else {
                        let mut si = Suggestion::empty();
                        // NOTE: this effectively clamps si_count to a certain minimum value, which it can't go below
                        let si_count: f64 = 10f64
                            / ((10i64).saturating_pow(term_list1[i].len() as u32))
                                as f64;

                        si.term = term_list1[i].clone();
                        si.count = si_count as i64;
                        si.distance = edit_distance_max + 1;
                        suggestion_parts.push(si);
                    }
                } else {
                    let mut si = Suggestion::empty();
                    // NOTE: this effectively clamps si_count to a certain minimum value, which it can't go below
                    let si_count: f64 = 10f64
                        / ((10i64).saturating_pow(term_list1[i].len() as u32)) as f64;

                    si.term = term_list1[i].clone();
                    si.count = si_count as i64;
                    si.distance = edit_distance_max + 1;
                    suggestion_parts.push(si);
                }
            }
        }

        let mut s = "".to_string();
        for si in suggestion_parts {
            s.push_str(&si.term);
            s.push(' ');
        }

        s.trim().to_string()
    }

    /// Divides a string into words by inserting missing spaces at the appropriate positions
    ///
    ///
    /// # Arguments
    ///
    /// * `input` - The word being segmented.
    /// * `max_edit_distance` - The maximum edit distance between input and suggested words.
    ///
    /// # Examples
    ///
    /// ```
    /// use compose::{SymSpell, Verbosity};
    ///
    /// let mut symspell = SymSpell::default();
    /// symspell.using_dictionary_file("data/frequency_dictionary_en_82_765.txt", 0, 1, " ");
    /// symspell.word_segmentation("itwas", 2);
    /// ```
    pub fn word_segmentation(&self, input: &str, max_edit_distance: i64) -> Composition {
        let input = ascii::prepare(input);
        let asize = input.len();

        let mut ci: usize = 0;
        let mut compositions: Vec<Composition> = vec![Composition::empty(); asize];

        for j in 0..asize {
            let imax = cmp::min(asize - j, self.max_length as usize);
            for i in 1..=imax {
                let top_prob_log: f64;

                let mut part: Cow<str> = Cow::Borrowed(ascii::slice(&input, j, j + i));

                let mut sep_len = 0;
                let mut top_ed: i64 = 0;

                let first_char = ascii::at(&part, 0).unwrap();
                if first_char.is_whitespace() {
                    part = Cow::Owned(ascii::remove(&part, 0));
                } else {
                    sep_len = 1;
                }

                top_ed += part.len() as i64;

                part = Cow::Owned(part.replace(" ", ""));

                top_ed -= part.len() as i64;

                let results = self.lookup(&part, Verbosity::Top, max_edit_distance);

                if !results.is_empty() && results[0].distance == 0 {
                    top_prob_log = (results[0].count as f64 / WORD_COUNT as f64).log10();
                } else {
                    top_ed += part.len() as i64;
                    top_prob_log = (10.0
                        / (WORD_COUNT as f64 * 10.0f64.powf(part.len() as f64)))
                    .log10();
                }

                let di = (i + ci) % asize;

                // set values in first loop
                if j == 0 {
                    compositions[i - 1] = Composition {
                        segmented_string: part.to_string(),
                        distance_sum: top_ed,
                        prob_log_sum: top_prob_log,
                    };
                } else if i == self.max_length
                    || (((compositions[ci].distance_sum + top_ed
                        == compositions[di].distance_sum)
                        || (compositions[ci].distance_sum + sep_len + top_ed
                            == compositions[di].distance_sum))
                        && (compositions[di].prob_log_sum
                            < compositions[ci].prob_log_sum + top_prob_log))
                    || (compositions[ci].distance_sum + sep_len + top_ed
                        < compositions[di].distance_sum)
                {
                    compositions[di] = Composition {
                        segmented_string: format!(
                            "{} {}",
                            compositions[ci].segmented_string, part
                        ),
                        distance_sum: compositions[ci].distance_sum + sep_len + top_ed,
                        prob_log_sum: compositions[ci].prob_log_sum + top_prob_log,
                    };
                }
            }

            if j != 0 {
                ci += 1;
            }

            ci = if ci == asize { 0 } else { ci };
        }
        compositions[ci].to_owned()
    }

    fn delete_in_suggestion_prefix(
        &self,
        delete: &str,
        delete_len: i64,
        suggestion: &str,
        suggestion_len: i64,
    ) -> bool {
        if delete_len == 0 {
            return true;
        }
        let suggestion_len = if PREFIX_LENGTH < suggestion_len {
            PREFIX_LENGTH
        } else {
            suggestion_len
        };
        let mut j = 0;
        for i in 0..(delete_len as isize) {
            let del_char = ascii::at(delete, i).unwrap();
            while j < suggestion_len
                && del_char != ascii::at(suggestion, j as isize).unwrap()
            {
                j += 1;
            }

            if j == suggestion_len {
                return false;
            }
        }
        true
    }

    fn has_different_suffix(
        &self,
        max_edit_distance: i64,
        input: &str,
        input_len: i64,
        candidate_len: i64,
        suggestion: &str,
        suggestion_len: i64,
    ) -> bool {
        // handles the short circuit of min_distance
        // assignment when first boolean expression
        // evaluates to false
        let min = if PREFIX_LENGTH - max_edit_distance == candidate_len {
            cmp::min(input_len, suggestion_len) - PREFIX_LENGTH
        } else {
            0
        };

        (PREFIX_LENGTH - max_edit_distance == candidate_len)
            && (((min - PREFIX_LENGTH) > 1)
                && (ascii::suffix(input, (input_len + 1 - min) as usize)
                    != ascii::suffix(suggestion, (suggestion_len + 1 - min) as usize)))
            || ((min > 0)
                && (ascii::at(input, (input_len - min) as isize)
                    != ascii::at(suggestion, (suggestion_len - min) as isize))
                && ((ascii::at(input, (input_len - min - 1) as isize)
                    != ascii::at(suggestion, (suggestion_len - min) as isize))
                    || (ascii::at(input, (input_len - min) as isize)
                        != ascii::at(suggestion, (suggestion_len - min - 1) as isize))))
    }
}

#[inline]
fn parse_words(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split_whitespace()
        .map(String::from)
        .collect()
}

fn edits_prefix(max_dictionary_edit_distance: i64, key: &str) -> HashSet<Cow<str>> {
    let mut hash_set = HashSet::new();

    let key_len = key.len() as i64;

    if key_len <= max_dictionary_edit_distance {
        hash_set.insert(Cow::Borrowed(""));
    }

    if key_len > PREFIX_LENGTH {
        let shortened_key = ascii::slice(key, 0, PREFIX_LENGTH as usize);
        hash_set.insert(Cow::Borrowed(shortened_key));
        edits(
            max_dictionary_edit_distance,
            shortened_key,
            0,
            &mut hash_set,
        );
    } else {
        hash_set.insert(Cow::Borrowed(key));
        edits(max_dictionary_edit_distance, key, 0, &mut hash_set);
    };

    hash_set
}

fn edits(
    max_dictionary_edit_distance: i64,
    word: &str,
    edit_distance: i64,
    delete_words: &mut HashSet<Cow<str>>,
) {
    let edit_distance = edit_distance + 1;
    let word_len = word.len();

    if word_len > 1 {
        for i in 0..word_len {
            let delete: Cow<str> = Cow::Owned(ascii::remove(word, i));

            if !delete_words.contains(&delete) {
                delete_words.insert(delete.clone());

                if edit_distance < max_dictionary_edit_distance {
                    edits(
                        max_dictionary_edit_distance,
                        &delete,
                        edit_distance,
                        delete_words,
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bench_correct() {
        let mut sym_spell = SymSpell::default();
        sym_spell.using_dictionary_file(
            "./data/frequency_dictionary_en_82_765.txt",
            0,
            1,
            " ",
        );

        let edit_distance_max: i64 = 2;
        let start = std::time::Instant::now();
        for _ in 0..1000 {
            let typo = "whereis th elove";
            let correction = "whereas the love";
            let results = sym_spell.lookup_compound(typo, edit_distance_max);
            assert_eq!(correction, results);

            let typo = "the bigjest playrs";
            let correction = "the biggest players";
            let results = sym_spell.lookup_compound(typo, edit_distance_max);
            assert_eq!(correction, results);

            let typo = "Can yu readthis";
            let correction = "can you read this";
            let results = sym_spell.lookup_compound(typo, edit_distance_max);
            assert_eq!(correction, results);

            let typo = "whereis th elove hehad dated forImuch of thepast who couqdn'tread in sixthgrade and ins pired him";
            let correction = "whereas the love head dated for much of the past who couldn't read in sixth grade and inspired him";
            let results = sym_spell.lookup_compound(typo, edit_distance_max);
            assert_eq!(correction, results);

            let typo = "in te dhird qarter oflast jear he hadlearned ofca sekretplan";
            let correction =
                "in the third quarter of last year he had learned of a secret plan";
            let results = sym_spell.lookup_compound(typo, edit_distance_max);
            assert_eq!(correction, results);

            let typo =
                "the bigjest playrs in te strogsommer film slatew ith plety of funn";
            let correction =
                "the biggest players in the strong summer film slate with plenty of fun";
            let results = sym_spell.lookup_compound(typo, edit_distance_max);
            assert_eq!(correction, results);

            let typo = "Can yu readthis messa ge despite thehorible sppelingmsitakes";
            let correction =
                "can you read this message despite the horrible spelling mistakes";
            let results = sym_spell.lookup_compound(typo, edit_distance_max);
            assert_eq!(correction, results);
        }
        println!("{:?} {:?}/iter", start.elapsed(), start.elapsed() / 1000);
    }

    #[test]
    fn test_lookup_compound() {
        let edit_distance_max = 2;
        let mut sym_spell = SymSpell::default();
        sym_spell.using_dictionary_file(
            "./data/frequency_dictionary_en_82_765.txt",
            0,
            1,
            " ",
        );

        let typo = "whereis th elove";
        let correction = "whereas the love";
        let results = sym_spell.lookup_compound(typo, edit_distance_max);
        assert_eq!(correction, results);

        let typo = "the bigjest playrs";
        let correction = "the biggest players";
        let results = sym_spell.lookup_compound(typo, edit_distance_max);
        assert_eq!(correction, results);

        let typo = "Can yu readthis";
        let correction = "can you read this";
        let results = sym_spell.lookup_compound(typo, edit_distance_max);
        assert_eq!(correction, results);

        let typo = "whereis th elove hehad dated forImuch of thepast who couqdn'tread in sixthgrade and ins pired him";
        let correction = "whereas the love head dated for much of the past who couldn't read in sixth grade and inspired him";
        let results = sym_spell.lookup_compound(typo, edit_distance_max);
        assert_eq!(correction, results);

        let typo = "in te dhird qarter oflast jear he hadlearned ofca sekretplan";
        let correction =
            "in the third quarter of last year he had learned of a secret plan";
        let results = sym_spell.lookup_compound(typo, edit_distance_max);
        assert_eq!(correction, results);

        let typo = "the bigjest playrs in te strogsommer film slatew ith plety of funn";
        let correction =
            "the biggest players in the strong summer film slate with plenty of fun";
        let results = sym_spell.lookup_compound(typo, edit_distance_max);
        assert_eq!(correction, results);

        let typo = "Can yu readthis messa ge despite thehorible sppelingmsitakes";
        let correction =
            "can you read this message despite the horrible spelling mistakes";
        let results = sym_spell.lookup_compound(typo, edit_distance_max);
        assert_eq!(correction, results);
    }

    #[test]
    fn test_word_segmentation() {
        let edit_distance_max = 2;
        let mut sym_spell = SymSpell::default();
        sym_spell.using_dictionary_file(
            "./data/frequency_dictionary_en_82_765.txt",
            0,
            1,
            " ",
        );

        let typo = "thequickbrownfoxjumpsoverthelazydog";
        let correction = "the quick brown fox jumps over the lazy dog";
        let result = sym_spell.word_segmentation(typo, edit_distance_max);
        assert_eq!(correction, result.segmented_string);

        let typo = "itwasabrightcolddayinaprilandtheclockswerestrikingthirteen";
        let correction =
            "it was a bright cold day in april and the clocks were striking thirteen";
        let result = sym_spell.word_segmentation(typo, edit_distance_max);
        assert_eq!(correction, result.segmented_string);

        let typo =
            "itwasthebestoftimesitwastheworstoftimesitwastheageofwisdomitwastheageoffoolishness";
        let correction = "it was the best of times it was the worst of times it was the age of wisdom it was the age of foolishness";
        let result = sym_spell.word_segmentation(typo, edit_distance_max);
        assert_eq!(correction, result.segmented_string);
    }
}
