#pragma once
#include <string>
#include <memory>
#include <sstream>

enum class PlacementType {
    SHARD,      
    REPLICATE,  
    PARTIAL     


class Placement {
public:
    virtual ~Placement() = default;
    
    virtual PlacementType type() const = 0;
    virtual std::string describe() const = 0;
    

    virtual std::shared_ptr<Placement> clone() const = 0;

    virtual bool equals(const Placement* other) const = 0;
};


class Shard : public Placement {
public:
    explicit Shard(int dim) : dim_(dim) {}
    
    PlacementType type() const override { 
        return PlacementType::SHARD; 
    }
    
    int dim() const { return dim_; }
    
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
    int dim_;  
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

class Partial : public Placement {
public:
    explicit Partial(const std::string& reduce_op = "sum") 
        : reduce_op_(reduce_op) {}
    
    PlacementType type() const override { 
        return PlacementType::PARTIAL; 
    }
    
    std::string reduce_op() const { return reduce_op_; }
    
    std::string describe() const override {
        std::ostringstream oss;
        oss << "Partial(reduce_op=\"" << reduce_op_ << "\")";
        return oss.str();
    }
    
    std::shared_ptr<Placement> clone() const override {
        return std::make_shared<Partial>(reduce_op_);
    }
    
    bool equals(const Placement* other) const override {
        if (other->type() != PlacementType::PARTIAL) return false;
        return reduce_op_ == static_cast<const Partial*>(other)->reduce_op_;
    }
    
private:
    std::string reduce_op_;  
};
