from web3 import Web3, HTTPProvider
from web3.middleware import geth_poa_middleware
import time
web3 = Web3(Web3.HTTPProvider('https://rpc-mainnet.matic.network'))  # this is the Matic network
print(web3.isConnected())

web3.middleware_onion.inject(geth_poa_middleware, layer=0)

abi3 = [{"inputs":[],"name":"get","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"x","type":"uint256"}],"name":"set","outputs":[],"stateMutability":"nonpayable","type":"function"}]
# contracte = web3.eth.contract('0xDc587838956cC1642c73EfeB03C4BE9247a7F163', abi = abi3)

contracte = web3.eth.contract(abi = abi3)
web3.eth.defaultAccount = '0xDc587838956cC1642c73EfeB03C4BE9247a7F163';

account_1 = "0xDc587838956cC1642c73EfeB03C4BE9247a7F163"
private_key1 = "e84b8aa199fc04a5ba5599910bfe70e742dab2772c334ed1af5d0566f9d8a9b5"
nonce1 = web3.eth.getTransactionCount(account_1)

tx0 = {'nonce': nonce1, 'gas': 2000000, 'gasPrice': web3.toWei('5', 'gwei')}  # it was a matter of specifying less gas!
tx1 = {'nonce': nonce1, 'to':'0x66bB3214Fbf021817E632B7E6483D1F2039281D4', 'gas': 2000000, 'gasPrice': web3.toWei('5', 'gwei')}  # it was a matter of specifying less gas!
# contracte_txn = contracte.functions.set(15)  # to store the data
# signed_tx = web3.eth.account.signTransaction(tx0, private_key1)
# tx_hash = web3.eth.sendRawTransaction(signed_tx.rawTransaction)

# tx_receipt = web3.eth.waitForTransactionReceipt(tx_hash)

# print(contracte.functions.get().buildTransaction(tx0))

contracte_txn = contracte.functions.set(15).buildTransaction(tx1)  # to store the data
print(contracte_txn)
signed_contr = web3.eth.account.signTransaction(contracte_txn, private_key1)
tx_hash_c = web3.eth.sendRawTransaction(signed_contr.rawTransaction)
print(tx_hash_c)



