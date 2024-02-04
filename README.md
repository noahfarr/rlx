RLX: Reinforcement Learning with MLX

RLX is a collection of Reinforcement Learning algorithms implemented in MLX, Apple's new Machine Learning framework. This project aims to leverage the unified memory capabilities of Apple's M series chips to enhance the performance and efficiency of these algorithms.

Prerequisites

Python 3.9 or later
Poetry for dependency management
An Apple device with an M-series chip
Installation

Clone the repository to your local machine:

bash
Copy code
git clone https://github.com/noahfarr/rlx.git
cd rlx
Install dependencies using Poetry:

bash
Copy code
poetry install
Structure

The project is organized into directories by algorithm. Each directory contains the implementation of a specific Reinforcement Learning algorithm, making the project modular and scalable. Here's an overview:

alg1/: Implementation of Algorithm 1
alg2/: Implementation of Algorithm 2
...
Usage

To run a specific algorithm, navigate to its directory and execute the main script. For example:

bash
Copy code
cd alg1
poetry run python main.py
Replace alg1 with the directory of the algorithm you wish to run.

Contributing

Contributions to RLX are welcome. To contribute, please follow these steps:

Fork the repository.
Create a new branch for your feature (git checkout -b feature/AmazingFeature).
Commit your changes (git commit -m 'Add some AmazingFeature').
Push to the branch (git push origin feature/AmazingFeature).
Open a pull request.
License

This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments

Special thanks to the MLX team for providing the framework.
This project is designed to run optimally on Apple's M series chips.
