import json
from web3 import Web3
from solc import compile_standard

compiled_sol = compile_standard({
     "language": "Solidity",
     "sources": {
         "Greeter.sol": {
             "content": '''
                 pragma solidity ^0.5.0;

                 contract Greeter {
                   string public greeting;

                   constructor() public {
                        greeting = 'Hello';
                   }

                   function setGreeting(string memory _greeting) public {
                      greeting = _greeting;
                   }

                   function greet() view public returns (string memory) {
                       return greeting;
                   }
                 }
               '''
         }
     },
     "settings":
         {
             "outputSelection": {
                 "*": {
                     "*": [
                         "metadata", "evm.bytecode"
                         , "evm.bytecode.sourceMap"
                     ]
                 }
             }
         }
 })
w3 = Web3(Web3.EthereumTesterProvider())