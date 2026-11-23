# Prompts

These are the exact system instructions used in the experiments, extracted verbatim from the notebooks. They are also reproduced in Appendix C of the paper.

| File | Used for | Model | Source notebook |
|---|---|---|---|
| `step1_video_to_script.txt` | 2-step pipeline, step 1: TikTok video → textual script | Gemini 1.5 Pro (`gemini-1.5-pro-002`) | `notebooks/1_video_to_script/gemini_video_to_script.ipynb` |
| `direct_video_to_values.txt` | Direct pipeline: video → values (Gemini^d) | Gemini 1.5 Pro | `notebooks/2_llm_value_extraction/gemini_direct_video2values.ipynb` |
| `step2_script_to_values.txt` | 2-step pipeline, step 2: script → values (Gemini^2s, Mistral^2s) | Gemini 1.5 Pro; Mistral 7B | `notebooks/2_llm_value_extraction/gemini_2step_script2values.ipynb` |

## How the prompts are used

- **System instruction:** each prompt is passed as the model's system instruction.
- **User turn:** the text `"Video: "` followed by the mp4 file (step 1 and the direct pipeline), or the script text (step 2).
- **Direct vs. 2-step:** the step-2 prompt is the direct prompt with "video" replaced by "script". The direct prompt also ends with one extra line: "There is always some value in the video, dont return empty answers."
- **Mistral^2s:** called through a hosted Mistral agent configured with the step-2 prompt, with JSON output enforced via `response_format={"type": "json_object"}`.
- **Safety settings:** step 1 ran with Gemini's safety filters set to `BLOCK_NONE` for harassment, hate speech, sexually explicit and dangerous content, so that scripts of videos containing (e.g.) pranks or mild profanity were not blocked. The value-extraction calls used the default safety settings.
- **Sampling parameters:** none were set, so the model defaults apply. Each TikTok was processed once.
