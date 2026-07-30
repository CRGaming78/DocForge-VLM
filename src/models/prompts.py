import random

SYSTEM_PROMPT = """You are a highly skilled document forensics expert specializing in detecting identity document forgery, tampering, and digital manipulation. Your task is to analyze the provided image of an identity document and determine whether it is 'AUTHENTIC' or 'TAMPERED'. Look for common signs of forgery such as mismatched fonts, unnatural edge artifacts, inconsistent lighting or shadows, misalignment, or traces of digital editing."""

USER_PROMPTS = [
    "Analyze this identity document for signs of forgery or tampering. Is it authentic or tampered?",
    "Examine this document image carefully. Determine if it is genuine or has been digitally manipulated.",
    "You are reviewing this identity document. Check for any signs of tampering, forgery, or digital manipulation.",
    "Please inspect this ID card for authenticity. Is it a real document or a forgery?",
    "Conduct a forensic analysis of this document image. Can you identify any tampering?",
    "Review this identity document. State whether you believe it is authentic or tampered.",
    "Look closely at the details of this document. Does it appear to be genuine or altered?",
    "Assess this ID for signs of digital alteration. Is it authentic or tampered?"
]

AUTHENTIC_RESPONSES = [
    "AUTHENTIC. After careful examination, this document appears to be genuine. The lighting, fonts, and alignment are consistent throughout, and there are no visible signs of digital tampering or forgery.",
    "AUTHENTIC. I have reviewed the document and found no evidence of manipulation. The text and images align properly without unnatural artifacts.",
    "AUTHENTIC. The identity document exhibits consistent physical characteristics. No mismatched fonts, digital artifacts, or inconsistent shadows were detected.",
    "AUTHENTIC. This document passes forensic checks. The edges, holograms (if visible), and text alignment are natural and consistent with a genuine ID.",
    "AUTHENTIC. There are no indications of digital alteration or physical tampering. All elements are coherent and natural."
]

TAMPERED_RESPONSES = [
    "TAMPERED. The document shows signs of digital manipulation. There are inconsistencies in the font styles and unnatural artifacts around the text.",
    "TAMPERED. I detected evidence of forgery. The lighting on the portrait does not match the background, and there are signs of splicing.",
    "TAMPERED. This document appears to be altered. There is visible misalignment in the text fields and unnatural blurriness indicating digital editing.",
    "TAMPERED. Forensic analysis reveals tampering. Edge artifacts and compression inconsistencies suggest that elements have been digitally pasted into the document.",
    "TAMPERED. The ID is not genuine. Anomalies such as mismatched shadows and irregular character spacing point strongly to digital forgery."
]

def format_conversation(image_path: str, label: int, prompt_idx: int = None) -> list[dict]:
    """
    Format a conversational prompt for Qwen2-VL.
    
    Args:
        image_path (str): Path to the image.
        label (int): 0 for AUTHENTIC, 1 for TAMPERED.
        prompt_idx (int, optional): Index of the user prompt to use. Defaults to a random index.
        
    Returns:
        list[dict]: A list of message dictionaries.
    """
    if prompt_idx is None:
        prompt_idx = random.randint(0, len(USER_PROMPTS) - 1)
        
    user_text = USER_PROMPTS[prompt_idx]
    
    response = random.choice(AUTHENTIC_RESPONSES) if label == 0 else random.choice(TAMPERED_RESPONSES)
    
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_text},
            ],
        },
        {
            "role": "assistant",
            "content": response
        }
    ]
    return messages

def parse_verdict(response: str) -> tuple[str, float]:
    """
    Extract the verdict and pseudo-confidence from the model's response.
    
    Args:
        response (str): The response string from the model.
        
    Returns:
        tuple[str, float]: Verdict ('AUTHENTIC' or 'TAMPERED') and a confidence score.
    """
    verdict = "UNKNOWN"
    if "AUTHENTIC" in response.upper():
        verdict = "AUTHENTIC"
    elif "TAMPERED" in response.upper():
        verdict = "TAMPERED"
        
    # In a real scenario, confidence might come from logits.
    # Here we return a dummy high confidence if a verdict is found.
    confidence = 0.95 if verdict != "UNKNOWN" else 0.0
    
    return verdict, confidence
