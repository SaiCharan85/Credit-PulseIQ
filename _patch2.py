"""Wire injection refusal and advice redirect into ask()."""
from pathlib import Path

p = Path("agents/qa.py")
s = p.read_text(encoding="utf8")

old = '''    context = memo_context(payload)
    system = SYSTEM_PROMPT
    if asks_for_advice(question):
        system = SYSTEM_PROMPT + REDIRECT_DIRECTIVE
    messages = ['''
new = '''    # Refused before the model sees it. Passing an injection through and
    # relying on the output guard would work only for the phrasings that guard
    # knows, and it would leave the reader with a blocked answer and no idea
    # why. Saying what happened is both safer and more useful.
    if question_is_injection(question):
        return Answer(text=INJECTION_NOTICE, allowed=True, reason="question_injection")

    context = memo_context(payload)
    system = SYSTEM_PROMPT
    if asks_for_advice(question):
        system = SYSTEM_PROMPT + REDIRECT_DIRECTIVE
    messages = ['''
assert old in s
s = s.replace(old, new)
p.write_text(s, encoding="utf8")
print("wired")
