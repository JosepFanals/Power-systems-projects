from web3 import Web3, HTTPProvider
# web3 = Web3(Web3.HTTPProvider('https://rpc.cheapeth.org/rpc'))  # this is the cheapeth network
# web3 = Web3(Web3.HTTPProvider('https://rpc-mumbai.matic.today/'))  # this is the Matic network
web3 = Web3(Web3.HTTPProvider('https://rpc-mainnet.matic.network'))  # this is the Matic network
print(web3.isConnected())


contracte = web3.eth.contract('0xDc587838956cC1642c73EfeB03C4BE9247a7F163', abi = '[{"inputs": [{"internalType": "uint256","name": "num","type": "uint256"}],"name": "store","outputs": [],"stateMutability": "nonpayable","type": "function"},{"inputs": [],"name": "retrieve","outputs": [{"internalType": "uint256","name": "","type": "uint256"}],"stateMutability": "view","type": "function"}]')


# tx0 = {'nonce': nonce1, 'to': account_1, 'value': web3.toWei(0.003, 'ether'), 'gas': 2000000, 'gasPrice': web3.toWei('5', 'gwei')}
# contracte_txn = contracte.functions.store(12129).buildTransaction()

account_1 = "0xDc587838956cC1642c73EfeB03C4BE9247a7F163"
account_2 = "0x01E42BEAa16c42ee7d9314e79Ac59d29D2866A60"
private_key1 = "e84b8aa199fc04a5ba5599910bfe70e742dab2772c334ed1af5d0566f9d8a9b5"
private_key2 = "042226bb084a511ff452bac5fc77cff12cad0f51dd3818ee7182ad930c185824"
nonce1 = web3.eth.getTransactionCount(account_1)
nonce2 = web3.eth.getTransactionCount(account_2)

# tx0 = {'nonce': nonce1, 'gas': 20000000, 'gasPrice': web3.toWei('10', 'gwei')}
tx0 = {'nonce': nonce1, 'gas': 500000, 'gasPrice': web3.toWei('1', 'gwei')}  # it was a matter of specifying less gas!
contracte_txn = contracte.functions.store(13).buildTransaction(tx0)  # to create the contract
# contracte_txn_retrieve = contracte.functions.retrieve().buildTransaction(tx0)

balanc = web3.eth.getBalance(account_1)
print(balanc * 1e-18, 'eth')

# tx1 = {'nonce': nonce1, 'to': account_1, 'value': web3.toWei(0.003, 'ether'), 'gas': 2000000, 'gasPrice': web3.toWei('5', 'gwei')}

signed_tx1 = web3.eth.account.signTransaction(tx0, private_key1)
# signed_tx1 = web3.eth.account.signTransaction(tx0, private_key2)

signed_tx = signed_tx1
print(signed_tx)

tx_hash = web3.eth.sendRawTransaction(signed_tx.rawTransaction)
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