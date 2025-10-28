
#pragma once
#include <string>

enum class AllReduceOp { SUM, MAX, MIN, PROD };
enum class ReduceScatterOp { SUM, MAX, MIN, PROD };

class DTensor;

void all_reduce_(DTensor& t, AllReduceOp op);
void broadcast_(DTensor& t, int root);
DTensor all_gather(const DTensor& t);
void reduce_scatter_(DTensor& t, ReduceScatterOp op);
