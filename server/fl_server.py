import socket
import pickle
import numpy as np
import threading
from sklearn.ensemble import RandomForestRegressor

NUM_CLIENTS = 5

class ClientThread(threading.Thread):
    def __init__(self, client_socket):
        threading.Thread.__init__(self)
        self.client_socket = client_socket
        self.client_params = None

    def run(self):
        self.client_params = pickle.loads(self.client_socket.recv(2000000))  # Receive client parameters
        print("On server:", self.client_params)

    def get_params(self):
        return self.client_params        

def aggregate_params(client_params_list):
    # Simple averaging of parameters
    return np.mean(client_params_list, axis=0)

def merge_params(client_params_list):
    random_forest = RandomForestRegressor(n_estimators=NUM_CLIENTS)
    random_forest.estimators_ = client_params_list
    print(random_forest)
    return random_forest

def main():
    global_params = [0, 0]  # Initial global parameters
    host = '127.0.0.1'
    port = 12345

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)

    print("Server listening...")

    received_clients = 0
    client_threads = []

    while received_clients < NUM_CLIENTS:
        client_socket, addr = server_socket.accept()
        print(f"Connection from {addr} has been established.")
        
        client_thread = ClientThread(client_socket)
        client_thread.start()
        client_threads.append(client_thread)

        received_clients += 1

    for client_thread in client_threads:
        client_thread.join()

    client_params_list = [client_thread.get_params() for client_thread in client_threads if client_thread.get_params() is not None]
    if len(client_params_list) < NUM_CLIENTS:
        print("Not all clients provided parameters.")

    # global_params = merge_params(client_params_list)  # Aggregate parameters from all clients
    global_params = merge_params(client_params_list)
    print("Updated global parameters:", global_params)

    for client_thread in client_threads:
        client_socket = client_thread.client_socket
        client_socket.send(pickle.dumps(global_params))
        client_socket.shutdown(socket.SHUT_WR)  # Shutdown sending side of the socket
    
    for client_thread in client_threads:
        client_socket = client_thread.client_socket
        client_socket.close()

    server_socket.close()

if __name__ == "__main__":
    main()
