# Example prompt override.
#
# bench.py applies a prompt override by importing this file and, for every
# UPPER/underscore name it defines that ALSO exists on `autorag.agent`,
# doing `setattr(autorag.agent, name, value)` *before* build_topic_runnable()
# is called (the agent reads these constants when it builds the chains).
#
# Rules for an override:
#   - Only redefine the constants you want to change. Leave the rest alone.
#   - Keep every Jinja-ish `{name}` template variable the agent fills in
#     (for _L1_SYS there are none; the human template owns {transcript},
#     {audio_e}, {duration_min}, {target_count}).
#   - Literal braces must stay doubled: `{{"s": "MM:SS"}}` renders as
#     `{"s": "MM:SS"}`. A single `{` will raise a KeyError at format time.
#   - The name must already exist on autorag.agent or it is ignored (with a
#     warning) — this guards against typos silently doing nothing.
#
# This example tightens _L1_SYS into a shorter, more directive prompt to test
# whether terseness changes boundary quality or token cost. It is a
# DEMONSTRATION, not a tuned win — measure it like any other variant.

_L1_SYS = (
    "You are a topic boundary detector. Input: a transcript as blocks "
    "separated by blank lines; every line is `MM:SS-MM:SS Speaker K: "
    "<words>`. Output: ordered, non-overlapping L1 topics that tile the "
    "whole recording.\n\n"
    "Constraints:\n"
    '1. Items are intervals only: {{"s": "MM:SS", "e": "MM:SS"}} -- no '
    "titles, no summaries.\n"
    "2. First topic s = first MM:SS in transcript; last topic e = last "
    "MM:SS in transcript.\n"
    "3. Siblings tile end-to-start (B.s = A.e), ordered by time.\n"
    "4. Hit roughly the suggested count; never 15+ tiny topics, never one "
    "topic unless the audio is very short.\n"
    "5. Copy MM:SS values verbatim from the range markers; never invent or "
    "reformat them."
)
