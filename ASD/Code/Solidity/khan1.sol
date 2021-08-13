pragma solidity >=0.7.0 <0.8.0;

contract EnergyTrade{
    struct Offer{
        uint32 ID;
        address seller;
        address buyer;
        string P2L_Hash;
        string L2P_Hash;
        uint32 energy;
        uint32 price;
        uint timeOffered;
        uint timePaidP2L;
        bool picked;
        bool P2L_Paid;
        bool L2P_Paid;
        bool delivered;
    }
    
    mapping(uint32 => Offer) public offers;
    uint32 public offerID;
    
    function addOffer(uint32 energy, uint32 price) public{
        Offer storage newOffer = offers[offerID];
        newOffer.ID = offerID;
        newOffer.seller = msg.sender;
        newOffer.energy = energy;
        newOffer.price = price;
        newOffer.timeOffered = block.timestamp;
        newOffer.picked = false;
        newOffer.P2L_Paid = false;
        newOffer.L2P_Paid = false;
        newOffer.delivered = false;
        offerID += 1;
    }
    
    function pickOffer(uint32 id) public {
        offers[id].picked = true;
    }
    
    function confirmP2L_Tx(uint32 id, string memory P2L_Hash) public {
        offers[id].P2L_Paid = true;
        offers[id].P2L_Hash = P2L_Hash;
        offers[id].buyer = msg.sender;
        offers[id].timePaidP2L = block.timestamp;
    }
    
    function PoD(uint32 id) public{
        offers[id].delivered = true;
    }
    
    function confirmL2P_Tx(uint32 id, string memory L2P_Hash) public {
        offers[id].L2P_Paid = true;
        offers[id].L2P_Hash = L2P_Hash;
    }
    
    
    
}

