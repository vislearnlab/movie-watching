One CSV per video (or per reviewer part), all read by pipeline/prepare_prompts.py.
Header (exact): video,kind,prompt_idx,prompt_name,sam_text,shot,start_s,end_s,frame,x,y,x0,y0,x1,y1,note
Lines starting with # are comments. Kinds are documented in the prepare_prompts.py docstring.
Validate with: python pipeline/prepare_prompts.py --check
