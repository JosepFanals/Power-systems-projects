import json
from web3 import Web3

w3 = Web3(Web3.HTTPProvider('https://rpc.cheapeth.org/rpc'))


account_1 = "0xDc587838956cC1642c73EfeB03C4BE9247a7F163"
private_key1 = 'e84b8aa199fc04a5ba5599910bfe70e742dab2772c334ed1af5d0566f9d8a9b5'

# w3.eth.defaultAccount = account_1
# account_0 = w3.eth.account.privateKeyToAccount('0x' + private_key1)
# print(account_0)
# w3.eth.defaultAccount = account_0
# w3.eth.accounts.wallet.add('0xe84b8aa199fc04a5ba5599910bfe70e742dab2772c334ed1af5d0566f9d8a9b5')
# w3.eth.defaultAccount = account_0.address


print(w3.eth.accounts.wallet)

abi = json.loads('[{"inputs": [{"internalType": "uint256","name": "num","type": "uint256"}],"name": "store","outputs": [],"stateMutability": "nonpayable","type": "function"},{"inputs": [],"name": "retrieve","outputs": [{"internalType": "uint256","name": "","type": "uint256"}],"stateMutability": "view","type": "function"}]')

bytecode = '608060405234801561001057600080fd5b5061012f806100206000396000f3fe6080604052348015600f57600080fd5b506004361060325760003560e01c80632e64cec11460375780636057361d146051575b600080fd5b603d6069565b6040516048919060c2565b60405180910390f35b6067600480360381019060639190608f565b6072565b005b60008054905090565b8060008190555050565b60008135905060898160e5565b92915050565b60006020828403121560a057600080fd5b600060ac84828501607c565b91505092915050565b60bc8160db565b82525050565b600060208201905060d5600083018460b5565b92915050565b6000819050919050565b60ec8160db565b811460f657600080fd5b5056fea264697066735822122062db17618d746a1967495ede611efc2c1e881cb29cbd6b40b23bd35a720c134c64736f6c63430008010033'

Greeter = w3.eth.contract(abi=abi, bytecode=bytecode)

tx_hash = Greeter.constructor().transact()
# tx_hash = Greeter.constructor()

print(tx_hash)