import warnings

# Deprecation warning for Box2D
warnings.warn(
    "The 'box2d' environments are deprecated and will be removed in a future version. "
    "Please migrate to the 'pymunk' versions of these environments.",
    DeprecationWarning,
    stacklevel=2,
)

# ... existing Box2D implementation code ...
