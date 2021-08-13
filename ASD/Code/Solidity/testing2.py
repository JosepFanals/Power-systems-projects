from web3 import Web3, HTTPProvider
# web3 = Web3(Web3.HTTPProvider('https://rpc.cheapeth.org/rpc'))  # this is the cheapeth network
web3 = Web3(Web3.HTTPProvider('https://rpc-mainnet.matic.network'))  # this is the Matic network
print(web3.isConnected())

account_1 = "0xDc587838956cC1642c73EfeB03C4BE9247a7F163"
account_2 = "0x01E42BEAa16c42ee7d9314e79Ac59d29D2866A60"
account_3 = "0x5e335D154A1515bcE3b237bBCDDca1E1398FA8C8"
account_4 = "0x69F8E212c97DB9Ec721f508a038a1BA724131946"

private_key1 = "e84b8aa199fc04a5ba5599910bfe70e742dab2772c334ed1af5d0566f9d8a9b5"
private_key2 = "042226bb084a511ff452bac5fc77cff12cad0f51dd3818ee7182ad930c185824"

nonce1 = web3.eth.getTransactionCount(account_1) 
nonce2 = web3.eth.getTransactionCount(account_2)

tx1 = {'nonce': nonce1, 'to': account_2, 'value': web3.toWei(0.000000003, 'ether'), 'gas': 21000, 'gasPrice': web3.toWei('1', 'gwei')}
# tx2 = {'nonce': nonce2, 'to': account_1, 'value': web3.toWei(0.0000003, 'ether'), 'gas': 2000000, 'gasPrice': web3.toWei('5', 'wei')}

signed_tx1 = web3.eth.account.signTransaction(tx1, private_key1)
# signed_tx2 = web3.eth.account.signTransaction(tx2, private_key2)

signed_tx = signed_tx1

tx_hash = web3.eth.sendRawTransaction(signed_tx.rawTransaction)
print(signed_tx)
print(tx_hash)
# tx_hash = web3.toHex(tx_hash)
# blockNumber = web3.eth.blockNumber
# blockHash = web3.eth.getBlock(blockNumber).hash
# blockHash = web3.toHex(blockHash)
# timestamp = web3.eth.getBlock(blockNumber).timestamp

# print(tx_hash)
# print(blockNumber)
# print(blockHash)
# print(timestamp)

