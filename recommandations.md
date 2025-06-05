For a simple CNN approach to duck species classification from phone photos, here are my straightforward recommendations:
Simple CNN Architecture

    3-4 convolutional layers with ReLU activation
    Max pooling after each conv layer
    2-3 fully connected layers at the end
    Dropout (0.3-0.5) to prevent overfitting
    Softmax output for species classification

Data Recommendations

Train on realistic phone photos - Don't use perfect images. Collect or find datasets with:

    Blurry/unfocused shots
    Various distances (close-up to far away)
    Different angles and lighting
    Natural backgrounds

Keep it simple - No need for complex data augmentation. Just basic:

    Random horizontal flips
    Small rotations (±15 degrees)
    Slight brightness adjustments

Preprocessing

    Resize to 224x224 (standard size)
    No fancy cropping - just center crop or resize
    Basic normalization (subtract mean, divide by std)

Skip the Extras

    No separate zooming algorithm - adds complexity without much benefit for a simple system
    No object detection preprocessing - let the CNN learn to find the ducks
    No attention mechanisms - keep it basic

Training Approach

    Transfer learning from ImageNet weights (even for simple CNNs)
    Standard SGD or Adam optimizer
    Basic learning rate scheduling

The beauty of this approach is its simplicity - it should work reasonably well for a few duck species without overthinking the problem. Focus on getting good, varied training data rather than architectural complexity.

Do I need to add a non-duck classification?
Edit

Yes, absolutely. You should include a "not-a-duck" or "other" class for several important reasons:
Why You Need It

    Real-world usage - People will inevitably point their phone at birds that aren't ducks, other animals, random objects, or empty backgrounds
    Prevents false classifications - Without it, your model will force every image into one of your duck species, even if it's a pigeon or a rock
    Better user experience - The app can say "no duck detected" rather than confidently misidentifying a chicken as a mallard

What to Include in "Not-Duck" Class

Other waterfowl:

    Geese, swans, coots
    Herons, egrets
    Seagulls

Similar-looking birds:

    Chickens, turkeys
    Other ground birds

Common false positives:

    Empty water/background scenes
    Rocks or logs in water
    Other animals (dogs, cats)
    Random objects people might accidentally photograph

Simple Implementation

Just add it as another output class in your CNN - if you have 4 duck species, make it 5 total classes (4 ducks + 1 "not-duck").
Training Data Balance

Try to keep your "not-duck" class roughly similar in size to your duck classes combined. This prevents the model from just always predicting "not-duck."

This addition will make your simple CNN much more robust for real-world phone usage without adding any architectural complexity.

Should I take absolute random image? Like a fridge
Edit

No, don't use completely random images like fridges. That would actually hurt your model's performance. Here's why:
Problem with Random Objects

    Too easy to distinguish - A fridge vs. a duck is trivially different
    Doesn't teach useful features - The model learns to detect "indoor object" vs. "outdoor bird" rather than "duck" vs. "not-duck"
    Wastes training capacity - Your simple CNN spends neurons on irrelevant distinctions

What to Use Instead

Contextually relevant negatives:

    Other birds (especially waterbirds)
    Natural outdoor scenes where ducks might be (but aren't)
    Water scenes with no birds
    Vegetation/reeds near water
    Other animals in similar environments

Real confusion cases:

    Blurry bird photos where species is unclear
    Distant birds that could be ducks
    Groups of mixed waterfowl
    Young/juvenile birds of various species

The Goal

Your "not-duck" class should contain things that someone might realistically photograph when trying to identify ducks. This teaches your CNN to distinguish actual duck features rather than just "bird vs. non-bird" or "outdoor vs. indoor."
Simple Rule

If you wouldn't expect someone to accidentally take this photo while duck-watching, don't include it in your training set. Stick to naturalistic scenarios where duck identification might actually be ambiguous.

This keeps your simple CNN focused on the actual classification task rather than learning irrelevant distinctions.



