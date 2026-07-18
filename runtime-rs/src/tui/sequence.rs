use std::collections::BTreeSet;
use std::fmt::{Display, Formatter};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SequenceParseError {
    pub message: String,
    pub byte: usize,
    pub column: usize,
}

impl Display for SequenceParseError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{} at column {}", self.message, self.column)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextBuffer {
    text: String,
    cursor: usize,
    anchor: Option<usize>,
}

impl Default for TextBuffer {
    fn default() -> Self {
        Self::new(String::new())
    }
}

impl TextBuffer {
    pub fn new(text: impl Into<String>) -> Self {
        let text = text.into();
        let cursor = text.chars().count();
        Self {
            text,
            cursor,
            anchor: None,
        }
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn cursor(&self) -> usize {
        self.cursor
    }

    pub fn byte_cursor(&self) -> usize {
        char_to_byte(&self.text, self.cursor)
    }

    pub fn selection(&self) -> Option<(usize, usize)> {
        let anchor = self.anchor?;
        (anchor != self.cursor).then_some({
            if anchor < self.cursor {
                (anchor, self.cursor)
            } else {
                (self.cursor, anchor)
            }
        })
    }

    pub fn set(&mut self, text: impl Into<String>) {
        self.text = text.into();
        self.cursor = self.text.chars().count();
        self.anchor = None;
    }

    pub fn select_all(&mut self) {
        self.anchor = Some(0);
        self.cursor = self.text.chars().count();
    }

    pub fn insert(&mut self, value: &str) {
        self.delete_selection();
        let byte = self.byte_cursor();
        self.text.insert_str(byte, value);
        self.cursor += value.chars().count();
    }

    pub fn backspace(&mut self) {
        if self.delete_selection() {
            return;
        }
        if self.cursor == 0 {
            return;
        }
        let end = self.byte_cursor();
        let start = char_to_byte(&self.text, self.cursor - 1);
        self.text.replace_range(start..end, "");
        self.cursor -= 1;
    }

    pub fn delete(&mut self) {
        if self.delete_selection() {
            return;
        }
        let char_count = self.text.chars().count();
        if self.cursor >= char_count {
            return;
        }
        let start = self.byte_cursor();
        let end = char_to_byte(&self.text, self.cursor + 1);
        self.text.replace_range(start..end, "");
    }

    pub fn move_left(&mut self, selecting: bool) {
        self.prepare_selection(selecting);
        self.cursor = self.cursor.saturating_sub(1);
        self.finish_selection(selecting);
    }

    pub fn move_right(&mut self, selecting: bool) {
        self.prepare_selection(selecting);
        self.cursor = (self.cursor + 1).min(self.text.chars().count());
        self.finish_selection(selecting);
    }

    pub fn move_home(&mut self, selecting: bool) {
        self.prepare_selection(selecting);
        self.cursor = 0;
        self.finish_selection(selecting);
    }

    pub fn move_end(&mut self, selecting: bool) {
        self.prepare_selection(selecting);
        self.cursor = self.text.chars().count();
        self.finish_selection(selecting);
    }

    fn prepare_selection(&mut self, selecting: bool) {
        if selecting && self.anchor.is_none() {
            self.anchor = Some(self.cursor);
        }
        if !selecting {
            self.anchor = None;
        }
    }

    fn finish_selection(&mut self, selecting: bool) {
        if selecting && self.anchor == Some(self.cursor) {
            self.anchor = None;
        }
    }

    fn delete_selection(&mut self) -> bool {
        let Some((start, end)) = self.selection() else {
            self.anchor = None;
            return false;
        };
        let start_byte = char_to_byte(&self.text, start);
        let end_byte = char_to_byte(&self.text, end);
        self.text.replace_range(start_byte..end_byte, "");
        self.cursor = start;
        self.anchor = None;
        true
    }
}

pub fn parse_layer_sequence(
    text: &str,
    available_layers: &BTreeSet<usize>,
) -> Result<Vec<usize>, SequenceParseError> {
    let bytes = text.as_bytes();
    let mut position = 0usize;
    skip_whitespace(bytes, &mut position);
    require_byte(
        text,
        bytes,
        &mut position,
        b'[',
        "Expected `[` to begin the sequence",
    )?;
    skip_whitespace(bytes, &mut position);
    if bytes.get(position) == Some(&b']') {
        return Err(parse_error(
            text,
            position,
            "The sequence must contain at least one layer",
        ));
    }

    let mut sequence = Vec::new();
    loop {
        skip_whitespace(bytes, &mut position);
        let number_start = position;
        while bytes.get(position).is_some_and(u8::is_ascii_digit) {
            position += 1;
        }
        if number_start == position {
            return Err(parse_error(
                text,
                position,
                "Expected a zero-based numeric layer index",
            ));
        }
        let layer = text[number_start..position].parse::<usize>().map_err(|_| {
            parse_error(text, number_start, "Layer index is too large to represent")
        })?;
        if !available_layers.contains(&layer) {
            return Err(parse_error(
                text,
                number_start,
                format!(
                    "Unknown layer `{layer}`. Available layers: {}",
                    format_available_layers(available_layers)
                ),
            ));
        }
        sequence.push(layer);
        skip_whitespace(bytes, &mut position);
        match bytes.get(position) {
            Some(b',') => position += 1,
            Some(b']') => {
                position += 1;
                break;
            }
            Some(_) => {
                return Err(parse_error(
                    text,
                    position,
                    "Expected `,` or `]` after the layer index",
                ));
            }
            None => {
                return Err(parse_error(
                    text,
                    position,
                    "Sequence is incomplete; add a closing `]`",
                ));
            }
        }
    }
    skip_whitespace(bytes, &mut position);
    if position != bytes.len() {
        return Err(parse_error(
            text,
            position,
            "Unexpected text after the closing `]`",
        ));
    }
    Ok(sequence)
}

fn require_byte(
    text: &str,
    bytes: &[u8],
    position: &mut usize,
    expected: u8,
    message: &str,
) -> Result<(), SequenceParseError> {
    if bytes.get(*position) != Some(&expected) {
        return Err(parse_error(text, *position, message));
    }
    *position += 1;
    Ok(())
}

fn skip_whitespace(bytes: &[u8], position: &mut usize) {
    while bytes.get(*position).is_some_and(u8::is_ascii_whitespace) {
        *position += 1;
    }
}

fn parse_error(text: &str, byte: usize, message: impl Into<String>) -> SequenceParseError {
    SequenceParseError {
        message: message.into(),
        byte,
        column: text[..byte.min(text.len())].chars().count() + 1,
    }
}

fn format_available_layers(layers: &BTreeSet<usize>) -> String {
    let mut ranges = Vec::new();
    let mut iter = layers.iter().copied();
    let Some(mut start) = iter.next() else {
        return "none".to_string();
    };
    let mut end = start;
    for layer in iter {
        if layer == end + 1 {
            end = layer;
        } else {
            ranges.push(format_range(start, end));
            start = layer;
            end = layer;
        }
    }
    ranges.push(format_range(start, end));
    ranges.join(", ")
}

fn format_range(start: usize, end: usize) -> String {
    if start == end {
        start.to_string()
    } else {
        format!("{start}-{end}")
    }
}

fn char_to_byte(text: &str, character: usize) -> usize {
    text.char_indices()
        .nth(character)
        .map(|(byte, _)| byte)
        .unwrap_or(text.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_zero_based_numeric_sequences_with_whitespace() {
        let available = [0, 1, 2, 3].into_iter().collect();
        assert_eq!(
            parse_layer_sequence(" [ 0, 1,2, 1 ] ", &available).unwrap(),
            vec![0, 1, 2, 1]
        );
    }

    #[test]
    fn reports_unknown_layer_at_precise_column() {
        let available = [0, 1, 2].into_iter().collect();
        let error = parse_layer_sequence("[0, 7, 2]", &available).unwrap_err();
        assert_eq!(error.byte, 4);
        assert_eq!(error.column, 5);
        assert!(error.message.contains("Available layers: 0-2"));
    }

    #[test]
    fn rejects_temporarily_incomplete_input_without_a_partial_sequence() {
        let available = [0, 1, 2].into_iter().collect();
        assert!(
            parse_layer_sequence("[0,1,", &available)
                .unwrap_err()
                .message
                .contains("Expected")
        );
    }

    #[test]
    fn text_buffer_supports_selection_replacement_and_unicode_cursor_positions() {
        let mut buffer = TextBuffer::new("abλ");
        buffer.move_left(true);
        buffer.move_left(true);
        buffer.insert("XY");
        assert_eq!(buffer.text(), "aXY");
        assert_eq!(buffer.cursor(), 3);
        assert_eq!(buffer.byte_cursor(), 3);
        buffer.select_all();
        buffer.backspace();
        assert_eq!(buffer.text(), "");
    }
}
