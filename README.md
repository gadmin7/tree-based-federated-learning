# Tree-Based Federated Learning
Basic Cross-Silo Horizontal Federated Learning PoC using Socket Programming with Decision Tree-based Model Training

# About
Federated Learning (FL) is a groundbreaking approach to collaborative machine learning that prioritizes data privacy by enabling multiple parties to train models without directly sharing raw data. This project focuses on a specific FL paradigm Horizontal Federated Learning.

In this proof of concept, we utilize Walmart's Weekly Sales data to simulate an environment with multiple clients and a server, all hosted locally. Each client represents a separate data silo, and their contributions are aggregated on a central server.

Our implementation employs Socket Programming, a fundamental communication mechanism, to facilitate interaction between distributed clients and the server.

# Run Commands
**Running Server**
- Open a terminal and navigate to the project directory. Run the following command to start the server: `python fl_server.py`
- The server will start and begin listening for client connections on `127.0.0.1:12345`.

**Running Client**
- Open multiple terminals (one for each client) and navigate to the project directory in each terminal. Run the following command in each terminal to start a client:`python fl_client.py`
- Ensure you start the same number of clients as specified in `NUM_CLIENTS` in fl_server.py. By default, this number is 5.
