// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace flexkv::gdr {
// Inputs are canonical sysfs paths, not BDF numbers or enumeration indices.
// Prefer the deepest common ancestor, then the fewest PCIe path hops.
inline std::pair<int,int> pci_proximity(const std::string &device,
                                       const std::string &nic) {
  if(device.empty() || nic.empty()) return {-1,0};
  std::vector<std::string> a,b;
  for(const auto &p:std::filesystem::path(device)) a.push_back(p.string());
  for(const auto &p:std::filesystem::path(nic)) b.push_back(p.string());
  size_t common=0;bool shared_pci_root=false;
  while(common<a.size() && common<b.size() && a[common]==b[common]) {
    if(a[common].rfind("pci",0)==0) shared_pci_root=true;
    ++common;
  }
  if(!shared_pci_root) return {-1,0};
  return {static_cast<int>(common),-static_cast<int>(a.size()+b.size()-2*common)};
}

inline size_t select_local_nic(const std::string &device,
    const std::vector<std::pair<std::string,std::string>> &nics) {
  std::pair<int,int> best{-1,0};size_t selected=0;bool tied=false;
  for(size_t i=0;i<nics.size();++i) {
    const auto score=pci_proximity(device,nics[i].second);
    if(score.first<0) continue;
    if(score>best) {best=score;selected=i;tied=false;}
    else if(score==best) tied=true;
  }
  if(best.first<0 || tied) {
    std::string message="GDR: ";
    message+=tied?"ambiguous nearest NIC":"no active NIC with a shared PCI root";
    message+=" for device path="+device+"; candidates:";
    for(const auto &n:nics) message+=" "+n.first+"="+n.second;
    throw std::runtime_error(message+"; check sysfs visibility or set gdr.nic_names explicitly");
  }
  return selected;
}
} // namespace flexkv::gdr
