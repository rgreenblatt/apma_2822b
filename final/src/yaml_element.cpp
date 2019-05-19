#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "yaml_element.hpp"

using namespace std;

YAML_Element::YAML_Element(const std::string &key_arg,
                           const std::string &value_arg) {
  key = key_arg;
  value = value_arg;
}

YAML_Element::~YAML_Element() {
  for (size_t i = 0; i < children.size(); i++) {
    delete children[i];
  }
  children.clear();
}

/*
 * Add an element to the vector
 * QUESTION: if an element is not added because the key already exists,
 * will this lead to memory leakage?
 */
template <typename T>
YAML_Element *YAML_Element::add(const std::string &key_arg, T value_arg) {
  this->value = "";
  string converted_value = convert_to_string(value_arg);
  YAML_Element *element = new YAML_Element(key_arg, converted_value);
  children.push_back(element);
  return element;
}

template YAML_Element *YAML_Element::add(const std::string &key_arg,
                                         int value_arg);
template YAML_Element *YAML_Element::add(const std::string &key_arg,
                                         long int value_arg);
template YAML_Element *YAML_Element::add(const std::string &key_arg,
                                         long long int value_arg);
template YAML_Element *YAML_Element::add(const std::string &key_arg,
                                         double value_arg);
template YAML_Element *YAML_Element::add(const std::string &key_arg,
                                         const std::string value_arg);
template YAML_Element *YAML_Element::add(const std::string &key_arg,
                                         const char * value_arg);
template YAML_Element *YAML_Element::add(const std::string &key_arg,
                                         float value_arg);
template YAML_Element *YAML_Element::add(const std::string &key_arg,
                                         unsigned long value_arg);

/*
 * returns pointer to the YAML_Element for the given key.
 * I, cam, believe an exception should be thrown if there is no
 * element in the vector for the specified key
 */
YAML_Element *YAML_Element::get(const std::string &key_arg) {
  for (size_t i = 0; i < children.size(); i++) {
    if (children[i]->getKey() == key_arg) {
      return children[i];
    }
  }
  return 0;
}

/*
 * prints a line of a YAML document.  Correct YAML depends on
 * correct spacing; the parameter space should be the proper
 * amount of space for the parent element
 */
string YAML_Element::printYAML(std::string space) {
  string yaml_line = space + key + ": " + value + "\n";
  for (int i = 0; i < 2; i++)
    space = space + " ";
  for (size_t i = 0; i < children.size(); i++) {
    yaml_line = yaml_line + children[i]->printYAML(space);
  }
  return yaml_line;
}

template <class T> 
string YAML_Element::convert_to_string(T value_arg) {
  stringstream strm;
  strm << value_arg;
  return strm.str();
}
