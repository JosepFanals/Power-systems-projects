from web3 import Web3, HTTPProvider
from web3.middleware import geth_poa_middleware
import time
# web3 = Web3(Web3.HTTPProvider('https://rpc.cheapeth.org/rpc'))  # this is the cheapeth network
# web3 = Web3(Web3.HTTPProvider('https://rpc-mumbai.matic.today/'))  # this is the Matic network
web3 = Web3(Web3.HTTPProvider('https://rpc-mainnet.matic.network'))  # this is the Matic network
print(web3.isConnected())

# web3.middleware_stack.inject(geth_poa_middleware, layer=0)
web3.middleware_onion.inject(geth_poa_middleware, layer=0)

# contracte = web3.eth.contract('0xDc587838956cC1642c73EfeB03C4BE9247a7F163', abi = '[{"inputs": [{"internalType": "uint256","name": "num","type": "uint256"}],"name": "store","outputs": [],"stateMutability": "nonpayable","type": "function"},{"inputs": [],"name": "retrieve","outputs": [{"internalType": "uint256","name": "","type": "uint256"}],"stateMutability": "view","type": "function"}]')

abi2 = [{"inputs": [], "name": "retrieve", "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"},{"inputs": [{"internalType": "uint256", "name": "num","type": "uint256"}],"name": "store","outputs": [],"stateMutability": "nonpayable","type": "function"}]
abi3 = [{"inputs":[],"name":"get","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"x","type":"uint256"}],"name":"set","outputs":[],"stateMutability":"nonpayable","type":"function"}]

# contracte = web3.eth.contract('0xDc587838956cC1642c73EfeB03C4BE9247a7F163', abi = abi2)
contracte = web3.eth.contract('0xDc587838956cC1642c73EfeB03C4BE9247a7F163', abi = abi3)
# contracte.web3.middleware_onion.inject(geth_poa_middleware, layer=0)
# web3.middleware_stack.inject(geth_poa_middleware, layer=0)


# tx0 = {'nonce': nonce1, 'to': account_1, 'value': web3.toWei(0.003, 'ether'), 'gas': 2000000, 'gasPrice': web3.toWei('5', 'gwei')}
# contracte_txn = contracte.functions.store(12129).buildTransaction()

account_1 = "0xDc587838956cC1642c73EfeB03C4BE9247a7F163"
account_2 = "0x01E42BEAa16c42ee7d9314e79Ac59d29D2866A60"
private_key1 = "e84b8aa199fc04a5ba5599910bfe70e742dab2772c334ed1af5d0566f9d8a9b5"
private_key2 = "042226bb084a511ff452bac5fc77cff12cad0f51dd3818ee7182ad930c185824"
nonce1 = web3.eth.getTransactionCount(account_1)
nonce2 = web3.eth.getTransactionCount(account_2)

# tx0 = {'nonce': nonce1, 'gas': 20000000, 'gasPrice': web3.toWei('10', 'gwei')}
tx0 = {'nonce': nonce1, 'gas': 2000000, 'gasPrice': web3.toWei('5', 'gwei')}  # it was a matter of specifying less gas!
# contracte_txn = contracte.functions.store(15).buildTransaction(tx0)  # to store the data
# contracte_txn = contracte.functions.set(15).buildTransaction(tx0)  # to store the data
contracte_txn = contracte.functions.set(15)  # to store the data
# contracte_txn_retrieve = contracte.functions.retrieve().buildTransaction(tx0)

print(contracte.functions.get().buildTransaction(tx0))

signed_tx = web3.eth.account.signTransaction(tx0, private_key1)
# print(signed_tx)

tx_hash = web3.eth.sendRawTransaction(signed_tx.rawTransaction)


tx_receipt = web3.eth.waitForTransactionReceipt(tx_hash)

# contracte_txn = contracte.functions.store(15).buildTransaction(tx0)  # to store the data
# contracte_txn = contracte.functions.store(15).transact()  # to store the data
# contracte_txn = contracte.functions.store(15)  # to store the data


print(contracte.functions.get().buildTransaction(tx0))

# contracte_txn = contracte.functions.store(15).buildTransaction(tx0)  # to store the data
contracte_txn = contracte.functions.set(15).buildTransaction(tx0)  # to store the data
# signed_contr = web3.eth.account.signTransaction(contracte_txn, private_key1)
# tx_hash_c = web3.eth.sendRawTransaction(signed_contr.rawTransaction)

print(contracte_txn)
# tx_receipt = web3.eth.waitForTransactionReceipt(contracte_txn)

print('before')
time.sleep(10)
print('after')

# print(contracte.functions.retrieve().call())
# print(contracte.functions.retrieve().buildTransaction(tx0))
print(contracte.functions.get().buildTransaction(tx0))




# tx_receipt = web3.eth.waitForTransactionReceipt(contracte_txn)
# print(contracte.functions.retrieve().call())

# tx_hash = web3.toHex(tx_hash)
# blockNumber = web3.eth.blockNumber
# blockHash = web3.eth.getBlock(blockNumber).hash
# blockHash = web3.toHex(blockHash)
# timestamp = web3.eth.getBlock(blockNumber).timestamp

# print(tx_hash)
# print(blockNumber)
# print(blockHash)
# print(timestamp)

# not enought miners, so sometimes i do not get txs