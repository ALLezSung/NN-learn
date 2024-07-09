# Neural Network Learning

###### ![python](https://badgen.net/badge/python/3.10/blue?icon=github) ![torch](https://badgen.net/badge/torch/2.0.1)  ![license](https://badgen.net/static/license/GPLv3)

## Introduction

 This personal repo was established by a beginner in **neural networks** to learn knowledge about artificial intelligence. The main parts of repo are projects for self-learning and code bases.

## Project Structure

The project is organized as follows:

- `database`: Contains the datasets used in the project, such as the MNIST dataset.
- `docs`: Documentation related to the project, including design docs and user guides.
- `models`: Stores trained model files.
- `src`: The source code of the project.
  - `nets`: Contains definitions of neural network models. For example, `simple.py` defines some simple deep neural networks.
  - `utils`: Includes utility functions for the project, such as dataset building and splitting in `nuts.py`.
  - `main.py`: The main script of the project, handling data loading, model definition, and the training process.
- `tests`: Contains test scripts to ensure the components of the project work as expected.

For more detailed information, please refer to the [`PROJECT_STRUCTURE.md`](./docs/PROJECT_STRUCTURE.md).

## Installation


To install the necessary dependencies and libraries for this project, please follow these steps:

1. Clone the repository to your local machine:
    ```
    git clone https://github.com/ALLezSung/NN-learn.git
    ```

2. Navigate to the project directory:
    ```
    cd NN-learn
    ```

3. Install the required dependencies using pip:
    ```
    pip install -r requirements.txt
    ```

4. Once the installation is complete, you are ready to utilize the repo.

Please note that this installation assumes you have Python and pip already installed on your machine. If not, please install them before proceeding.

## Contributing

Contributions to the development of the repo are welcome. If you would like to contribute, please follow these guidelines:

1. Fork the repository on GitHub.
2. Create a new branch for your contribution.
3. Make your changes and commit them to your branch.
4. Push your branch to your forked repository.
5. Submit a pull request with a detailed description of your changes.

By contributing, you agree to license your contributions under the same license as the project.

We appreciate your contributions and look forward to your involvement in the development of the repo.

## License

This project is licensed under the GNU General Public License version 3 (GPLv3). For more details, please see the [LICENSE](LICENSE) file.