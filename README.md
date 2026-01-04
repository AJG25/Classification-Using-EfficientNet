Classification Using EfficientNet

## Project Overview
This project develops a multiclass image classification model that combines pre-trained image features from EfficientNet-B2 with structured attribute data to improve classification accuracy.

## Technologies
- Python
- torch (PyTorch framework for model building and training)
- timm (pre-trained models, e.g., EfficientNet)
- torchvision (image transformations & preprocessing)
- Pandas & NumPy (data manipulation)
- TensorFlow/Keras (image loading and resizing)
- matplotlib & seaborn (data visualization)

## Dataset
- Training Data: 3,926 images (up-sampled to 7,000 for class balance)
- Test Data: 4,000 images
- Classes: 200
- Attribute features: 94-dimensional vector per image

## Methodology
- **Data Preprocessing:** Resized images to a consistent size; up-sampled minority classes to address imbalance.  
- **Feature Engineering:** Attribute features reduced to 128 dimensions via a fully connected layer, then concatenated with image features.  
- **Model Architecture:**
  1. **EfficientNet-B2** for image feature extraction (pre-trained on ImageNet)  
  2. **Attribute Layer**: processes attribute features  
  3. **Concatenation**: merges image and attribute features  
  4. **Final Layer**: outputs multi-class predictions  

## Results & Key Takeaways
- Test accuracy: 66%  
- Attribute features improved performance over image-only baseline  
- Demonstrated skills in multimodal neural networks, handling class imbalance, and transfer learning
