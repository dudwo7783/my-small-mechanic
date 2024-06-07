import re

        
def contains_image_name(text):
    # pattern = re.compile(r'^(?=.*[A-Z])(?=.*\d)[A-Z\d]+$')
    pattern = re.compile(r'^(?=.*[A-Z])(?=.*\d)[A-Z\d]+$')
    pattern2 = re.compile(r'^(?=.*[A-Z])(?=.*\d)+$')
    matched = (bool(pattern.match(text))) or (bool(pattern2.match(text)))
    return matched

def contains_alpha_numeric(text):
    text = re.sub(r'\d+', '', text)
    return bool(re.search('[가-힣a-zA-Z0-9]', text))

def contains_korean(text):
    return bool(re.search('[가-힣]', text))
    
def contains_numeric(text):
    return bool(re.match(r'^[0-9!@#$%^&*()\-_=+[\]{};:\'",.<>?`~\\|/]*$', text))
    
def contains_image_name(text):
    # pattern = re.compile(r'^(?=.*[A-Z])(?=.*\d)[A-Z\d]+$')
    pattern = re.compile(r'^[A-Z\d-]+$')

    return bool(pattern.match(text))
