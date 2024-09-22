<h1>Face Recognition Using Transfer Learning</h1>

<p>
This project implements a real-time face recognition system using transfer learning. The pre-trained <strong>VGG16</strong> model is fine-tuned on a custom dataset, and the trained model is used to analyze and recognize faces in real time with <strong>OpenCV</strong>.
</p>

<h2>Project Overview</h2>
<p>
The goal of this project is to create an efficient face recognition model by leveraging <strong>transfer learning</strong>. We use the <strong>VGG16</strong> model, pre-trained on the <strong>ImageNet</strong> dataset, to extract high-level features from the images. A custom classification head is added to the model, and it is trained on our face dataset. Once trained, the model is saved for later use.
</p>
<p>
Using <strong>OpenCV</strong>, the model is integrated with a real-time video feed (e.g., from a webcam) to perform live face recognition.
</p>

<h2>Key Features</h2>
<ul>
  <li><strong>Transfer Learning</strong>: Utilizes the pre-trained VGG16 model, avoiding training from scratch and improving training efficiency.</li>
  <li><strong>Custom Training</strong>: The model is trained on a custom dataset containing labeled images of faces.</li>
  <li><strong>Real-Time Face Recognition</strong>: Implements OpenCV to perform face detection and recognition on live video.</li>
  <li><strong>Model Persistence</strong>: The trained model is saved as an HDF5 file (.h5) and can be reused without retraining.</li>
</ul>

<h2>Workflow</h2>

<h3>Data Preparation:</h3>
<ul>
  <li>Organize the dataset into a <strong>Train</strong> and <strong>Test</strong> directory, where each subdirectory represents a class (person).</li>
  <li>Apply data augmentation techniques to make the model more robust.</li>
</ul>

<h3>Transfer Learning:</h3>
<ul>
  <li>Load the pre-trained <strong>VGG16</strong> model without the top layers.</li>
  <li>Add a custom fully connected layer for face classification.</li>
  <li>Train the model on the custom face dataset using <strong>categorical_crossentropy</strong> loss and <strong>adam</strong> optimizer.</li>
</ul>

<h3>Model Saving:</h3>
<ul>
  <li>After training, save the trained model as <strong>facefeatures_new_model.h5</strong> to avoid re-training.</li>
</ul>

<h3>Real-Time Face Recognition:</h3>
<ul>
  <li>Use <strong>OpenCV</strong> to capture live video from a webcam.</li>
  <li>Detect faces using OpenCV’s built-in Haar cascade or other detection methods.</li>
  <li>Preprocess the detected face, feed it to the trained model, and classify it in real time.</li>
</ul>
