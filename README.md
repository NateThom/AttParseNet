# AttParseNet

Welcome to the AttParseNet code repository! This work is the result of a research project completed by Nathan Thom and 
 Emily Hand at the University of Nevada, Reno - Machine Perception Laboratory.<br/><br/>

**If this work is a benefit to your own efforts, please cite our paper here:** <br/><br/>

![AttParseNet Architecture](https://github.com/natethom/AttParseNet/blob/master/README_Images/AttParseNet.png?raw=true)
Example of AttParseNet Architecture <br/><br/>

AttParseNet is a simple convolutional neural network for facial attribute recognition. It is unique and novel because it
 combines the tasks of facial attribute recognition (predicting which attributes are present in a given facial image) 
 and facial semantic segmentation (labeling each pixel in an image where an attribute occurs). The beauty of this
 approach is that attribute prediction accuracy is increased by asking the network to tell us which attributes are 
 occurring and where they are occurring. The segmentation task is only used during training. At run time no segments are
 used. <br/>
 
**Here's how it works:**<br/>
<ul>
    <li>Collect or download a dataset with facial attributes labeled (We use Celeba: 
        http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
    </li>
    <li>
        Run automatic facial landmark detection software to collect 64 landmark points from each image in your dataset.
        We use:
        <ul>
            <li>
                The OpenCV DLib landmark detector: http://dlib.net/
            </li>
            <li>
                OpenFace: https://github.com/TadasBaltrusaitis/OpenFace
            </li>
            <li>
                A hand-annotation tool: https://github.com/NateThom/Face-Annotation-Tool
            </li>
        </ul>
    </li>
    <li>
        Create semantic segmentation labels for each input image/attribute pair in your dataset. If you're using CelebA
        this will result in 8,104,000 labels (202,600 input images, each with 40 attribute labels -> 202,600 * 40 = 8,104,000)
        <ul>
            <li>
                Semantic segmentation labels are single channel, black and white images
            </li>
            <li>
                Black (pixel value of 0) denotes any pixels where the attribute does not occur
            </li>
            <li>
                White (pixel value of 255) denotes regions where the attribute does occur
            </li>
        </ul>
    </li>
    <li>
        Train a CNN with a joint learning architecture (i.e. two loss functions)
        <ul>
            <li>
                Our two loss functions are Binary Cross Entropy with Logits (attribute prediction loss) and Mean Squared
                Error (segmentation loss)
            </li>
            <li>
                Simply calculate both loss values and sum them
            </li>
            <li>
                You can use a CNN of whatever complexity you desire. We use a fairly vanilla architecture with 6 
                convolution layers, 1 pooling layer, and 1 fully connected layer
            </li>
        </ul>
    </li>
</ul>

![Segment Label Example](https://github.com/natethom/AttParseNet/blob/master/README_Images/Example_Segment_Label.jpg?raw=true)

Example of segment label

**What is in this repository:**<br/>
<ul>
    <li>
        attparsenet.py
        <ul>
            <li>
                PyTorch implementation
            </li>
            <li>
                Reads in data, trains a new or pretrained model, tests the model
            </li>
        </ul>
    </li>
    <li>
        attparsenet_landmark_labeler.py
        <ul>
            <li>
                Generates and stores the facial landmark labels to csv
            </li>
            <li>
                Uses the opencv and openface to automatically detect 68 facial landmarks
            </li>
        </ul>
    </li>
    <li>
        attparsenet_regions.py
        <ul>
            <li>
                Helper functions for forming foundational regions of the face from facial landmarks
            </li>
            <li>
                These regions are used in "attparsenet_segments.py" to form the regions where each attribute occurs
            </li>
        </ul>
    </li>
    <li>
        attparsenet_segment_labeler.py
        <ul>
            <li>
                Generates and stores the attribute segment label images
            </li>
            <li>
                Uses the "attparsenet_segments.py" file to form the regions where each attribute occurs on a given face
            </li>
        </ul>
    </li>
    <li>
        attparsenet_segments.py
        <ul>
            <li>
                Helper functions for forming the segments of a face where each attribute occurs
            </li>
            <li>
                Uses the "attparsenet_regions.py" file to form the segments where each attribute occurs on a given face
            </li>
            <li>
                These segments are used in "attparsenet_segment_labeler.py" to generate the segment label images
            </li>
        </ul>
    </li>
    <li>
        attparsenet_utils.py
        <ul>
            <li>
                Python argparse file
            </li>
            <li>
                Stores helpful configuration items (file paths, number of training epochs, etc.) in one easy to access place
            </li>
        </ul>
    </li>
</ul>

![Facial Landmark Layout](https://github.com/natethom/AttParseNet/blob/master/README_Images/Landmarks.jpeg?raw=true)
Layout of the 68 facial landmarks