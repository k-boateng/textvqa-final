def build_prompt(question, ocr_tokens=None, prompt_type="naive"):
    """
    Build a prompt for TextVQA inference.

    Args:
        question: TextVQA question.
        ocr_tokens: Optional list of OCR tokens from the dataset.
        prompt_type: One of ["naive", "ocr_aware", "constrained"].

    Returns:
        Prompt string.
    """
    if ocr_tokens is None:
        ocr_tokens = []

    # Keep OCR prompt short enough to avoid wasting context.
    ocr_text = ", ".join([str(tok) for tok in ocr_tokens[:80]])

    if prompt_type == "naive":
        return (
            "Answer the question based on the image. "
            "Give a short answer only.\n"
            f"Question: {question}"
        )

    if prompt_type == "ocr_aware":
        return (
            "Answer the question based on the image and the detected OCR text. "
            "Use the OCR text only if it is relevant. "
            "Give a short answer only.\n"
            f"OCR text: {ocr_text}\n"
            f"Question: {question}"
        )

    if prompt_type == "constrained":
        return (
            "You are answering a TextVQA question. "
            "The answer is usually a short word, number, brand, name, or phrase visible in the image. "
            "Do not explain. Do not answer in a full sentence. "
            "Return only the final answer.\n"
            f"OCR text: {ocr_text}\n"
            f"Question: {question}"
        )

    raise ValueError(f"Unknown prompt_type: {prompt_type}")