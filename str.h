/* $Id$ */

#ifndef STR_H
#define STR_H

#include <cstring>
#include <cstdlib>
#include <iostream>
using namespace std;
#include <cstdio>
#include <ctype.h>
#include "die.h"

class Str
{
  struct StrRep
  {
    int ref;
    char *p;
    
    StrRep(const char *s) : 
      ref(1), p(new char[strlen(s)+1]) 
    { strcpy(p, s); }
    
    ~StrRep() { if(p) { delete [] p; p = 0; } }
  };
  
  StrRep *rep;

  void destroy() 
  { 
    if (--rep->ref == 0) { 
      if(rep) { delete rep; rep = 0; }
    }
  }
  
public:

  operator char*() { return rep->p; }

  operator char*() const { return rep->p; }
  
  operator const char*() const { return rep->p; }

  operator const char*() { return rep->p; }

  ~Str() { destroy(); }

  Str() : rep(new StrRep('\0')) { }

  Str(const Str &s) : rep(s.rep) { rep->ref++; }

  Str(const char *s) : rep(new StrRep(s)) { }
  
  Str(int i)
  {
    char buf[64];
    sprintf(buf,"%d",i);
    rep = new StrRep(buf);
  }
  
  Str(double f)
  {
    char buf[64];
    sprintf(buf, "%f", f);
    rep = new StrRep(buf);
  }
  
  Str & operator =(const Str &s)
  {
    s.rep->ref++;
    destroy();
    rep = s.rep;
    return *this;
  }
  
  Str & operator =(const char *v) { return *this = Str(v); }
  
  Str copy() const { return Str((const char *) (*this)); }
  
  friend istream & operator >>(istream &c, Str &s)
  {
    char q, buf[1024];
    char *b = buf;
    c >> q;
    if (q == '"') { // quoted string
      while(c.get(*b) && *b != '"')
	b++;
    } else {        // unquoted string
      c.putback(q);
      while(c.get(*b) && !isspace(*b))
	b++;
    }
    if (buf - b > 1023)
      die("Str: string too long\n");
    *b = '\0';
    s = buf;
    return c;
  }
  
  friend ostream & operator <<(ostream &c, const Str &s)
  { return c << (const char *) s; }

  friend Str operator +(const Str &s1, const Str &s2)
  {
    char *buf = new char[strlen(s1) + strlen(s2) + 1];
    strcpy(buf, s1);
    strcat(buf, s2);
    Str s(buf);
    if(buf) { delete [] buf; buf = 0; }
    return s;
  }

  Str & operator +=(const char *s)
  {
    char *buf = new char[strlen(*this) + strlen(s) + 1];
    strcpy(buf, *this);
    strcat(buf, s);
    *this = buf;
    delete[] buf;
    buf = 0;
    return *this;
  }

  friend Str operator +(const char *s1, const Str &s2)
  { return Str(s1) + s2; }

  friend Str operator +(const Str &s1, const char *s2)
  { return s1 + Str(s2); }

  friend int operator <(const Str &s1, const Str &s2)
  { return strcmp(s1, s2) < 0; }

  friend int operator >(const Str &s1, const Str &s2)
  { return strcmp(s1, s2) > 0; }

  friend const Str &max(const Str &s1, const Str &s2)
  { return s1 > s2 ? s1 : s2; }

  friend const Str &min(const Str &s1, const Str &s2)
  { return s1 < s2 ? s1 : s2; }
};

#endif /* STR_H */
