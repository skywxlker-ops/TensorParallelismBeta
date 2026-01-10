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
    virtual ~Placement(); // Declaration only, defined in placement.cpp

    virtual PlacementType type() const = 0;
    virtual int dim() const { return -1; } 
    virtual void setDim(int) { } 
    virtual std::string describe() const = 0;
    virtual std::shared_ptr<Placement> clone() const = 0;
    virtual bool equals(const Placement* other) const = 0;
};


class Shard : public Placement {
public:
    Shard() = default;
    
    explicit Shard(int dim) : dim_(dim) {}
    
    PlacementType type() const override { 
        return PlacementType::SHARD; 
    }
    
    int dim() const override { return dim_; }
    void setDim(int dim) override { dim_ = dim; }
    
    std::string describe() const override {
        std::ostringstream oss;
        oss << "Shard(dim=" << dim_ << ")";
        return oss.str();
    }
    
    std::shared_ptr<Placement> clone() const override {
        return std::make_shared<Shard>(dim_);
    }
    
    bool equals(const Placement* other) const override {
        if (other->type() != PlacementType::SHARD) return false;
        return dim_ == static_cast<const Shard*>(other)->dim_;
    }

private:
    int dim_ = 0;  
};


class Replicate : public Placement {
public:
    Replicate() = default;
    
    PlacementType type() const override { 
        return PlacementType::REPLICATE; 
    }
    
    std::string describe() const override {
        return "Replicate()";
    }
    
    std::shared_ptr<Placement> clone() const override {
        return std::make_shared<Replicate>();
    }
    
    bool equals(const Placement* other) const override {
        return other->type() == PlacementType::REPLICATE;
    }
};
