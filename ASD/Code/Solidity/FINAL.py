from web3 import Web3, HTTPProvider
from web3.middleware import geth_poa_middleware
import time
web3 = Web3(Web3.HTTPProvider('https://rpc-mainnet.matic.network'))  # this is the Matic network
print(web3.isConnected())

web3.middleware_onion.inject(geth_poa_middleware, layer=0)

abi3 = [{"inputs":[],"name":"get","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"x","type":"uint256"}],"name":"set","outputs":[],"stateMutability":"nonpayable","type":"function"}]
# contracte = web3.eth.contract('0xDc587838956cC1642c73EfeB03C4BE9247a7F163', abi = abi3)

contracte = web3.eth.contract(abi = abi3)
# web3.eth.defaultAccount = '0xDc587838956cC1642c73EfeB03C4BE9247a7F163';

account_1 = "0xDc587838956cC1642c73EfeB03C4BE9247a7F163"
private_key1 = "e84b8aa199fc04a5ba5599910bfe70e742dab2772c334ed1af5d0566f9d8a9b5"
nonce1 = web3.eth.getTransactionCount(account_1)

tx1 = {'nonce': nonce1, 'to':'0x66bB3214Fbf021817E632B7E6483D1F2039281D4', 'gas': 2000000, 'gasPrice': web3.toWei('5', 'gwei')}  # it was a matter of specifying less gas!


contracte_txn = contracte.functions.set(15121998).buildTransaction(tx1)  # to store the data
signed_contr = web3.eth.account.signTransaction(contracte_txn, private_key1)
tx_hash_c = web3.eth.sendRawTransaction(signed_contr.rawTransaction)


# contracte_ret = contracte.functions.get()  # to retrieve the data. It is public, so this does not do anyting!
# contracte_ret = contracte.functions.get().buildTransaction(tx1)  # to retrieve the data
# signed_ret =  web3.eth.account.signTransaction(contracte_ret, private_key1)
# tx_hash_ret = web3.eth.sendRawTransaction(signed_ret.rawTransaction)



