#pragma once
#include <string>
#include <memory>
#include <sstream>

enum class PlacementType {
    SHARD,      
    REPLICATE    
};


class Placement {
public:
    Placement() = default;
    virtual ~Placement() = default;
    // Placement() = default;
    virtual PlacementType type() const = 0;
    virtual int dim() const {} // temp optimization

};


class Shard : public Placement {
public:
    explicit Shard(int dim) : dim_(dim) {}
    Shard() = default;
    
    PlacementType type() const override { 
        return PlacementType::SHARD; 
    }
    
    
    int dim() const{ return dim_; }
    void setDim(int dim) { dim_ = dim; }
    
    
    
private:
    int dim_;  
};


class Replicate : public Placement {
public:
    Replicate() = default;
    
    PlacementType type() const override { 
        return PlacementType::REPLICATE; 
    }
    
};
  
