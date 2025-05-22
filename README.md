# cGPT with complex numbers

This project implements a complex GPT model.
when run output is printed by the first trained input, i.e., data[0] created randomly.

## Project Structure

```
cGPT-1
├── src
│   ├── main.py          # Entry point for the application
│   ├── model
│   │   └── gpt.py      # Defines the GPT model architecture
│   ├── data
│   │   └── dataset.py   # Handles data loading and preprocessing
│   ├── train
│   │   └── train.py     # Contains the training loop
│   └── utils
│       └── helpers.py   # Utility functions for logging and evaluation
├── requirements.txt      # Lists project dependencies
└── README.md             # Project documentation
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd cgpt-1
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the training process, execute the following command:
```
python src/main.py
```

## Overview

This project is designed to provide a simple yet effective implementation of a small GPT model using complex numbers.
