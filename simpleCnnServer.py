import flwr as fl

# Start Flower server

fl.server.start_server(config=fl.server.ServerConfig(num_rounds=3))