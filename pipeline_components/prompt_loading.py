########################
# Prompt for Appraisal #
########################
def build_appraisal_prompt(text, appraisal_q):
    """
    :input: test Reddit post (for inference), appraisal_q (for one dimension)
    :output: zero-shot prompt for step 1 (eliciting appraisals)
    """
    return f"""[Text] {text}

[Question] {appraisal_q} Please provide your answer in the following format: <likert>[]</likert><rationale>[]</rationale>. Your response should be concise and brief."""


####################################
# Prompt for Iterative-Reappraisal #
####################################
revision_prompt = """please revise the reappraisal response to additionally address this feedback, while minimally modifying the original response"""

def build_iterative_step_baseline(post, prev_step):
    return f"""[Text] {post}\n\n[Reappraisal Response] {prev_step}\n[Feedback] Please revise the reappraisal response to help the narrator reappraise the situation better. Your response should be concise and brief."""

def build_iterative_step_baseline_guideline(post, guidance, prev_step):
    return f"""[Text] {post}\n\n[Reappraisal Response] {prev_step}\n[Feedback] {guidance} Taking this into account, {revision_prompt}. Your response should be concise and brief."""

def build_iterative_step_w_appraisal(post, appraisal, prev_step):
    return f"""[Text] {post}\n\n[Reappraisal Response] {prev_step}\n[Feedback] {appraisal}. Based on the above appraisal, {revision_prompt}. Your response should be concise and brief."""