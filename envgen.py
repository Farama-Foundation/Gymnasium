import os
import json
from groq import Groq
import cairo
import math

# Initialize Groq client
groq = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Make sure dataset directory exists
output_dir = "dataset"
os.makedirs(output_dir, exist_ok=True)

# System prompt from LLM.py
system_prompt = """
You are an expert in python programming and the pycairo library. Your goal is to help a pair of agents, engaging in a referential game to develop an artificial language capable of communicating aspects of a vast and diverse set of attributes. You will be provided with a list of classes off symbols the agents can already identify well, and classes that the agents have failed to learn. Your task is to analyze the current communication capabilities of the agent and write python code using the pycairo library to suggest a new symbol that the agents should learn to communicate.

The suggested symbol must be:-
1) Learnable: A task that is not too difficult for the agent based on its current communication level. 
2) Feasible: should be a symbol drawable within the scope of the pycairo library
3) Novel: Not a symbol that already exists in the set provided
4) Compositional : Symbol construction may reuse familiar geometric primitives like colors/shapes as long aas they are visually different from the existing ones
5) Diversity: symbols are not restricted to closed geometric figures; symbols may also incorporate strokes and patterns achievable through pycairo. Our goal eventually is to create languages able to communicate a variety of patterns.

Note: The code you generate must save images in the specific folder requested in the prompt.
"""

# Example code from c.py to provide as reference

c_py_example = """import cairo
import os

output_dir = "dataset"
os.makedirs(output_dir, exist_ok=True)

# Red color
red = (1.0, 0.0, 0.0)

# Image and square parameters
filename = os.path.join(output_dir, "red_square.png")
surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 200, 200)
context = cairo.Context(surface)

# White background
context.set_source_rgb(1.0, 1.0, 1.0)
context.paint()

# Red square (centered, size 100x100)
size = 100
x = (200 - size) / 2
y = (200 - size) / 2
context.set_source_rgb(*red)
context.rectangle(x, y, size, size)
context.fill()

surface.write_to_png(filename)

print(f"Generated a single red square image in the '{output_dir}' directory.")
"""

# Provide abstract syntax examples from your notebook
pycairo_syntax_reference = """
Here are some examples of valid pycairo syntax to help you understand the API, use them to understand how to use curves, gradients, and styling.

Example 1: Complex Curves & Transparency
```python
context.set_line_width(0.04)
# Curve (x1, y1, x2, y2, x3, y3)
context.move_to(0.1, 0.5)
context.curve_to(0.4, 0.9, 0.6, 0.1, 0.9, 0.5)
context.stroke()

context.set_source_rgba(1, 0.2, 0.2, 0.6) # Red with 60% opacity
context.move_to(0.1, 0.5)
context.line_to(0.4, 0.9)
context.stroke()
```

Example 2: Radial and Linear Gradients
```python
# Linear Gradient
pat = cairo.LinearGradient(0.0, 0.0, 0.0, 1.0)
pat.add_color_stop_rgba(1, 0, 0, 0, 1) # Black
pat.add_color_stop_rgba(0, 1, 1, 1, 1) # White
context.rectangle(0, 0, 1, 1)
context.set_source(pat)
context.fill()

# Radial Gradient
pat = cairo.RadialGradient(0.45, 0.4, 0.1, 0.4, 0.4, 0.5)
pat.add_color_stop_rgba(0, 1, 1, 1, 1)
pat.add_color_stop_rgba(1, 0, 0, 0, 1)
context.set_source(pat)
context.arc(0.5, 0.5, 0.3, 0, 2 * math.pi)
context.fill()
```

Example 3: Rotated Ellipses Using Transformations
```python
# Pycairo draws ellipses by scaling the context before drawing an arc
context.save()
context.translate(100, 100) # Move to center (xc, yc)
context.rotate(math.pi / 4) # Rotate by 45 degrees
context.scale(1.0, 0.3)     # Scale width by 1x and height by 0.3x
context.arc(0.0, 0.0, 50, 0.0, 2.0 * math.pi)
context.restore() # Always restore context after scaling for ellipses
context.fill()
```

Example 4: Relative Paths with Fill Preservation
```python
context.move_to(50, 20)
context.rel_line_to(40, 40)  # Move 40px right, 40px down from current point
context.rel_line_to(-40, 40) # Move 40px left, 40px down from current point
context.close_path()         # Connects back to the first point automatically

context.set_source_rgb(0, 0, 1)
context.fill_preserve()      # Fills the shape but keeps the path active
context.set_source_rgb(0, 0, 0)
context.set_line_width(2)
context.stroke()             # Outline the filled shape
```

Example 5: Line Styles and Alpha Blending Groups
```python
# Line Caps & Joins
context.set_line_cap(cairo.LINE_CAP_ROUND) # Alternatives: cairo.LINE_CAP_BUTT, cairo.LINE_CAP_SQUARE
context.set_line_join(cairo.LINE_JOIN_BEVEL) # Alternatives: cairo.LINE_JOIN_MITER, cairo.LINE_JOIN_ROUND

# Alpha Transparency on a group of shapes
context.push_group()
context.rectangle(60, 60, 80, 80)
context.set_source_rgb(1, 0, 0)
context.fill()
context.pop_group_to_source()
context.paint_with_alpha(0.5) # Renders the entire group at 50% opacity
```
"""

# Track previously generated and learned titles
learned_titles = ['red square','blue circle']

# Data generation loop
num_iterations = 20

for iteration in range(num_iterations):
    print(f"\n--- Starting Iteration {iteration + 1} ---")

    iter_dir = f"dataset/iter-{iteration + 2}"
    os.makedirs(iter_dir, exist_ok=True)

    # Formulate the prompt
    user_prompt = f"""
Here is an example of code using pycairo to generate a dataset:
```python
{c_py_example}
```

{pycairo_syntax_reference}

The agents have currently successfully learned the following symbols:
{learned_titles if learned_titles else "None so far."}
The agents have failed to learn: None

Please reason about what visual concepts the agents should learn next.
Output a JSON object with 'reasoning', 'symbol name', and 'code' to generate the next batch of images,smbol name is what the symbol is actually. Remember to put the images in the '{iter_dir}' directory.
"""

    messages = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "user", "content": user_prompt})
    
    try:
        response = groq.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "python_code_generation",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "reasoning": {"type": "string"},
                            "symbol name": {"type": "string"},
                            "code": {"type": "string"}
                        },
                        "required": ["reasoning", "code", "symbol name"],
                        "additionalProperties": False
                    }
                }
            }
        )
        
        # Parse output
        result_content = response.choices[0].message.content or "{}"
        result = json.loads(result_content)
        
        title = result.get("symbol name", f"dataset_iter_{iteration}")
        reasoning = result.get("reasoning", "")
        code_to_run = result.get("code", "")
        
        print(f"Title: {title}")
        print(f"Reasoning: {reasoning}")
        
        # Save input, title, reasoning, and code to a text file in the iteration folder
        log_path = os.path.join(iter_dir, "info.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("=== INPUT PROMPT ===\n")
            f.write(user_prompt)
            f.write("\n\n=== TITLE ===\n")
            f.write(title)
            f.write("\n\n=== REASONING ===\n")
            f.write(reasoning)
            f.write("\n\n=== CODE ===\n")
            f.write(code_to_run)

        if code_to_run:
            print("Executing generated code...")
            # Execute the code to generate the dataset items
            # Using global namespace allows it to access cairo/math if it forgets imports
            exec(code_to_run, globals())
            print("Code execution finished.")
            # Store title in the list of learned patterns
            learned_titles.append(title)
        else:
            print("No code provided by the LLM.")
            
    except Exception as e:
        print(f"An error occurred during iteration {iteration + 1}: {e}")
        
print("\n--- Data Generation Loop Completed ---")
print("Learned titles over the loop:", learned_titles)
