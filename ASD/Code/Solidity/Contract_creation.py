from web3 import Web3, HTTPProvider
from web3.middleware import geth_poa_middleware
import time
web3 = Web3(Web3.HTTPProvider('https://rpc-mainnet.matic.network'))  # this is the Matic network
print(web3.isConnected())

web3.middleware_onion.inject(geth_poa_middleware, layer=0)

abi2 = [{"inputs": [], "name": "retrieve", "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"},{"inputs": [{"internalType": "uint256", "name": "num","type": "uint256"}],"name": "store","outputs": [],"stateMutability": "nonpayable","type": "function"}]
abi3 = [{"inputs":[],"name":"get","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"x","type":"uint256"}],"name":"set","outputs":[],"stateMutability":"nonpayable","type":"function"}]
abi_full = [{"inputs":[{"internalType":"uint32","name":"energy","type":"uint32"},{"internalType":"uint32","name":"price","type":"uint32"}],"name":"addOffer","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"get","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"proposalID","outputs":[{"internalType":"uint32","name":"","type":"uint32"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint32","name":"","type":"uint32"}],"name":"proposals","outputs":[{"internalType":"uint32","name":"ID","type":"uint32"},{"internalType":"address","name":"seller","type":"address"},{"internalType":"uint32","name":"energy","type":"uint32"},{"internalType":"uint32","name":"price","type":"uint32"},{"internalType":"uint256","name":"timeProposed","type":"uint256"}],"stateMutability":"view","type":"function"}]

# contracte = web3.eth.contract('0xDc587838956cC1642c73EfeB03C4BE9247a7F163', abi = abi3)
contracte = web3.eth.contract('0xDc587838956cC1642c73EfeB03C4BE9247a7F163', abi = abi_full)

account_1 = "0xDc587838956cC1642c73EfeB03C4BE9247a7F163"
account_2 = "0x01E42BEAa16c42ee7d9314e79Ac59d29D2866A60"
private_key1 = "e84b8aa199fc04a5ba5599910bfe70e742dab2772c334ed1af5d0566f9d8a9b5"
private_key2 = "042226bb084a511ff452bac5fc77cff12cad0f51dd3818ee7182ad930c185824"
nonce1 = web3.eth.getTransactionCount(account_1)
nonce2 = web3.eth.getTransactionCount(account_2)

tx0 = {'nonce': nonce1, 'gas': 2000000, 'gasPrice': web3.toWei('5', 'gwei')}  # it was a matter of specifying less gas!
# contracte_txn = contracte.functions.set(15)  # to store the data


signed_tx = web3.eth.account.signTransaction(tx0, private_key1)
tx_hash = web3.eth.sendRawTransaction(signed_tx.rawTransaction)
tx_receipt = web3.eth.waitForTransactionReceipt(tx_hash)

contracte_txn = contracte.functions.get().buildTransaction(tx0)  # to initialize contract 

print(signed_tx.hash)

