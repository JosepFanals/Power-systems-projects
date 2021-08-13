    # Pablo Borao, Josep Fanals

from web3 import Web3, HTTPProvider
from web3.middleware import geth_poa_middleware
import time
web3 = Web3(Web3.HTTPProvider('https://rpc-mainnet.matic.network'))
print(web3.isConnected())

web3.middleware_onion.inject(geth_poa_middleware, layer=0)  # avoid errors

# load smart contract
abi_full = [{"inputs":[{"internalType":"uint32","name":"energy","type":"uint32"},{"internalType":"uint32","name":"price","type":"uint32"}],"name":"addOffer","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"proposalID","outputs":[{"internalType":"uint32","name":"","type":"uint32"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint32","name":"","type":"uint32"}],"name":"proposals","outputs":[{"internalType":"uint32","name":"ID","type":"uint32"},{"internalType":"address","name":"seller","type":"address"},{"internalType":"uint32","name":"energy","type":"uint32"},{"internalType":"uint32","name":"price","type":"uint32"},{"internalType":"uint256","name":"timeProposed","type":"uint256"}],"stateMutability":"view","type":"function"}]
contract_x = web3.eth.contract(abi = abi_full)

# set accounts
account_1 = "0xDc587838956cC1642c73EfeB03C4BE9247a7F163"
account_2 = "0x01E42BEAa16c42ee7d9314e79Ac59d29D2866A60"
account_3 = "0x5e335D154A1515bcE3b237bBCDDca1E1398FA8C8"
account_4 = "0xC19Bf8141a295d356d9667bac06a1d3D259e99ea"
account_DSO = "0x7B83a155F88aC066Ac89e3d34ee9966Dd5710A26"
private_key_1 = "e84b8aa199fc04a5ba5599910bfe70e742dab2772c334ed1af5d0566f9d8a9b5"
private_key_2 = "042226bb084a511ff452bac5fc77cff12cad0f51dd3818ee7182ad930c185824"
private_key_3 = "ddb023c55a2fde460dd15ba86aa68430270af7a0e289c8dde8e157f25573e754"
private_key_4 = "7c78014e96767d7994cd9c9a9601a0b2418ed4663ac1586324dc6a65d32ec701"
private_key_DSO = "91ccaaf3a77c479db66e54d0f1eb137350cbea95d4a4be1bac4f624c982aaf03"

# choose account to transact
account = account_DSO
private_key = private_key_DSO

# call the smart contract
nonce = web3.eth.getTransactionCount(account)
tx1 = {'nonce': nonce, 'to':'0xD58E1ED59876e742Fe56C40b59Cc3942c02B9a98', 'gas': 2000000, 'gasPrice': web3.toWei('5', 'gwei')} 
contracte_txn = contract_x.functions.addOffer(8043, 0).buildTransaction(tx1) 
signed_contr = web3.eth.account.signTransaction(contracte_txn, private_key)
tx_hash_c = web3.eth.sendRawTransaction(signed_contr.rawTransaction)



# contracte_ret = contracte.functions.get()  # to retrieve the data. It is public, so this does not do anyting!
# contracte_ret = contracte.functions.get().buildTransaction(tx1)  # to retrieve the data
# signed_ret =  web3.eth.account.signTransaction(contracte_ret, private_key1)
# tx_hash_ret = web3.eth.sendRawTransaction(signed_ret.rawTransaction)




