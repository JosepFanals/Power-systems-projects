from web3 import Web3

ganache_url = "https://127.0.0.1:8545"
web3 = Web3(Web3.HTTPProvider(ganache_url))
w3 = Web3(Web3.HTTPProvider("https://127.0.0.1:8545"))

print(w3.isConnected())


# account_1 = "0xDc587838956cC1642c73EfeB03C4BE9247a7F163"
# account_2 = "0x01E42BEAa16c42ee7d9314e79Ac59d29D2866A60"

# private_key = "042226bb084a511ff452bac5fc77cff12cad0f51dd3818ee7182ad930c185824"

# nonce = web3.eth.getTransactionCount(account_1)

# tx = {'nonce': nonce, 'to': account_2, 'value': web3.toWei(0.1, 'ether'), 'gas': 2000000, 'gasPrice': web3.toWei('5', 'gwei')}

# signed_tx = web3.eth.account.signTransaction(tx, private_key)
# tx_hash = web3.eth.sendRawTransaction(signed_tx.rawTransaction)
# tx_hash = web3.toHex(tx_hash)
# print(tx_hash)