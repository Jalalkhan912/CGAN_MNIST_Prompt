# MNIST Digit Generator with Text Prompts  

## Overview  
The **MNIST Digit Generator with Text Prompts** is an innovative AI-powered tool designed to generate handwritten digits based on natural language inputs. It leverages a **DCGAN model** to produce high-quality digit images and **NLP techniques** to extract meaningful information from user text prompts. This project demonstrates the synergy between **Natural Language Processing (NLP)** and **Generative Adversarial Networks (GANs)**.  

---

## Features  
- **Text-based Requests**: Users can input text prompts like “Generate 5 images of digit 1” or “make 3 images of digit 7”  
- **Digit Generation**: The tool uses a DCGAN model trained on the **MNIST dataset** to generate synthetic images resembling human handwriting.  
- **Prompt Interpretation**: The system extracts the **digit** and **quantity** from the input using NLP.  
- **Interactive Output**: Generated digits are displayed instantly, providing a fun and visual experience.  

---

## Model Details  
- **DCGAN (Deep Convolutional GAN)**: A generative model that creates realistic MNIST digits through adversarial training.  
- **NLP Techniques**: Simple parsing methods extract the digit to generate and the number of instances from user input.  
- **Training Data**: The GAN model was trained on the **MNIST dataset**, containing 70,000 grayscale images of handwritten digits (0-9).  

---

## How It Works  
1. **Input Text Prompt**: Users provide a natural language input (e.g., “Generate 7 images of digit 4”).  
2. **NLP Parsing**: The input is parsed to identify the requested digit and quantity.  
3. **Generate Digits**: The DCGAN model generates the requested number of digit images.  
4. **Display Output**: The generated digits are shown in real-time on the screen.  

---

## Use Cases  
- **AI Art and Creativity**: Generate personalized digit-based artworks.  
- **Handwriting Simulation**: Demonstrate GANs' capability to mimic human handwriting.  
- **AI Learning Tools**: Serve as a fun and interactive tool for students learning AI concepts.  

---

## Future Improvements  
- **Advanced NLP**: Use more sophisticated NLP models (like GPT) to handle complex prompts.  
- **Web Interface**: Develop an intuitive web-based UI for users to interact with the tool easily.  
- **Multi-Digit Generation**: Enable generation of sequences of multiple different digits from a single prompt.  

---
