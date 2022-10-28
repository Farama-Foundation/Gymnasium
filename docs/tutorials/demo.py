"""
Demo tutorial script
=========================

This file is not listed in the website and serves only to give an example of a tutorial file. And is mostly a copy-paste from sphinx-gallery.
"""

# %%
# This is a section header
# ------------------------
# This is the first section!
# The `#%%` signifies to Sphinx-Gallery that this text should be rendered as
# rST and if using one of the above IDE/plugin's, also signifies the start of a
# 'code block'.

# This line won't be rendered as rST because there's a space after the last block.
myvariable = 2
print(f"my variable is {myvariable}")
# This is the end of the 'code block' (if using an above IDE). All code within
# this block can be easily executed all at once.

# %%
# This is another section header
# ------------------------------
#
# In the built documentation, it will be rendered as rST after the code above!
# This is also another code block.

print(f"my variable plus 2 is {myvariable + 2}")
