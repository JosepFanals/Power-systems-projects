from web3 import Web3, HTTPProvider
# web3 = Web3(Web3.HTTPProvider('https://rpc.cheapeth.org/rpc')) 
web3 = Web3(Web3.HTTPProvider('https://kovan.infura.io/v3/dc603f9e71324c94b1c685826369fa1d')) 
abi = '[{"inputs":[],"name":"decimals","outputs":[{"internalType":"uint8","name":"","type":"uint8"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"description","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint80","name":"_roundId","type":"uint80"}],"name":"getRoundData","outputs":[{"internalType":"uint80","name":"roundId","type":"uint80"},{"internalType":"int256","name":"answer","type":"int256"},{"internalType":"uint256","name":"startedAt","type":"uint256"},{"internalType":"uint256","name":"updatedAt","type":"uint256"},{"internalType":"uint80","name":"answeredInRound","type":"uint80"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"latestRoundData","outputs":[{"internalType":"uint80","name":"roundId","type":"uint80"},{"internalType":"int256","name":"answer","type":"int256"},{"internalType":"uint256","name":"startedAt","type":"uint256"},{"internalType":"uint256","name":"updatedAt","type":"uint256"},{"internalType":"uint80","name":"answeredInRound","type":"uint80"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"version","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"}]' 
# addr = '0x01E42BEAa16c42ee7d9314e79Ac59d29D2866A60' 
addr = '0x9326BFA02ADD2366b30bacB125260Af641031331'
contract = web3.eth.contract(address=addr, abi=abi) 
output_info = contract
# print(output_info.version().call())
latestData = contract.functions.latestRoundData().call() 
print(latestData)