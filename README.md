# Mindful AI

**Mindful AI** is an intelligent meditation recommendation system that suggests meditations tailored to the user’s chosen theme or prompt. It’s built using **FastAPI** and **machine learning** to promote mental wellness through personalized, theme-based meditation recommendations.

## Features

 **Personalized Meditation Suggestions**  
Get meditation recommendations based on themes, moods, or prompts you provide.

 **FastAPI-Powered Backend**  
Fast, scalable REST API for handling requests and delivering recommendations.

 **Machine Learning Integration**  
Leverages machine learning models to understand user input and choose suitable meditations.

 **Tests Included**  
Automated tests provided to help maintain correctness and reliability.

## Tech Stack

- **Python** – Core language  
- **FastAPI** – Web framework for building APIs  
- **Machine Learning** – Recommendation logic  
- **PyTest** – Testing framework


## Getting Started

These instructions will help you run Mindful AI locally for development and testing.

### Prerequisites

Make sure you have:

- Python 3.8+ installed
- A virtual environment tool like `venv`

### Install Dependencies

- git clone https://github.com/sushmavankhede24/Mindful_AI.git
- cd Mindful_AI

# create and activate virtual environment
- python -m venv venv
- source venv/bin/activate  # macOS/Linux
- venv\Scripts\activate     # Windows
  
- pip install -r requirements.txt

## Run the API
- uvicorn app.main:app --reload

The API will be served at http://127.0.0.1:8000

## Run Tests
- pytest

## Configuration
Adjust application settings and environment variables as needed in your local .env or system environment.

## To Contribute
Contributions are welcome! To contribute:
  - Fork the repository
  - Create a new branch: git checkout -b feature/YourFeature
  - Make your changes
  - Commit your changes: git commit -m "Add some feature"
  - Push your branch: git push origin feature/YourFeature
  - Open a Pull Request

## Licence
This project is open-source and available under the **MIT License**.

## Contact
Have questions or suggestions? Reach out or open a GitHub issue!

**Happy meditating! **🧘‍♂️
