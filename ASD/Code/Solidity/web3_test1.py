from web3 import Web3

w3 = Web3(Web3.HTTPProvider('https://rpc.cheapeth.org/rpc'))
# w3 = Web3(Web3.HTTPProvider('https://mainnet.infura.io/v3/dc603f9e71324c94b1c685826369fa1d'))
print(w3.isConnected())

aaa = w3.eth.getBlock(w3.eth.blockNumber)
print(aaa.number)