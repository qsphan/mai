#!/usr/bin/env python3
"""Reports every `*)` that Rocq skips because it sits inside a string that a
`"` opened inside a comment -- the sites of the
`comment-terminator-in-string` warning, which the build turns into an error.

  run: python3 tools/comment_quote_check.py [FILE_OR_DIR ...]
       (default: iris vtest-rocq model-xv6iris kernel-rocq user-rocq)

WHY: inside a comment Rocq lexes `"..."` as a string, and a `*)` inside that
string does not close the comment.  In a boxed comment a quotation that wraps
from one box line to the next therefore swallows the first line's `*)` and the
second line's `(*`.  Today the second line's `"` closes the string again and the
comment ends where it looks like it does -- but one quotation edited out of
balance and the comment runs on until some later comment's quote, and the error
(`Unterminated string` / `Unterminated comment`) lands far from the cause
(claude-notes/durable-notes.md).  Keep each quotation on one line.

The lexer mirrors Rocq's: `(*` nests, `*)` closes, `"` opens a string in which
`""` is an escaped quote and neither `(*` nor `*)` counts.  Strings OUTSIDE
comments are skipped the same way.  Exit status is 0 when nothing is found.
"""
import os
import sys

DEFAULT = ['iris', 'vtest-rocq', 'model-xv6iris', 'kernel-rocq', 'user-rocq']


def sites(text):
    """Yield (line, col) of each `*)` Rocq skips inside a comment string."""
    i, n, depth, line, bol = 0, len(text), 0, 1, 0
    while i < n:
        c = text[i]
        if c == '"':
            j = i + 1
            while j < n:
                if text[j] == '"':
                    if j + 1 < n and text[j + 1] == '"':
                        j += 2
                        continue
                    break
                if text[j] == '\n':
                    line, bol = line + 1, j + 1
                elif depth and text.startswith('*)', j):
                    yield line, j - bol
                j += 1
            i = j + 1
            continue
        if c == '\n':
            line, bol = line + 1, i + 1
        elif text.startswith('(*', i):
            depth += 1
            i += 2
            continue
        elif depth and text.startswith('*)', i):
            depth -= 1
            i += 2
            continue
        i += 1


def files(args):
    for a in args:
        if os.path.isdir(a):
            for root, _, names in os.walk(a):
                for f in sorted(names):
                    if f.endswith('.v'):
                        yield os.path.join(root, f)
        elif a.endswith('.v'):
            yield a


def main():
    found = 0
    for path in files(sys.argv[1:] or [d for d in DEFAULT if os.path.isdir(d)]):
        with open(path, encoding='utf-8') as fh:
            text = fh.read()
        for line, col in sites(text):
            print(f'{path}:{line}:{col}: `*)` inside a quotation in a comment')
            found += 1
    print(f'{found} site(s)', file=sys.stderr)
    return 1 if found else 0


if __name__ == '__main__':
    sys.exit(main())
