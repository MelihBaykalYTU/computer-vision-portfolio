/* New AI-assisted educational companion, 2026-10-08.
 * Inspired by a 2024 joint report by Ceyda Tolunay and Melih Baykal.
 * NOT recovered original code or a reproduction of historical results. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

typedef struct { int n, d; double *x, *y; } Data;
typedef struct { double loss, accuracy; } Scores;
static uint32_t state;
static void die(const char *s) { fprintf(stderr,"%s\n",s); exit(1); }
static uint32_t rnd(void) {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    return state;
}
static double *alloc(size_t n) {
    double *p=calloc(n,sizeof(double)); if (!p) die("allocation failed"); return p;
}
static Data load(const char *path) {
    FILE *f=fopen(path,"r"); Data a={0,0,NULL,NULL}; char line[32768];
    if (!f) die("cannot open dataset");
    if (!fgets(line,sizeof(line),f) || sscanf(line,"%d,%d",&a.n,&a.d)!=2 ||
        a.n<1 || a.n>200000 || a.d<1 || a.d>2000) die("bad dataset header");
    a.d += 1;
    a.x=alloc((size_t)a.n*(size_t)a.d); a.y=alloc((size_t)a.n);
    for(int i=0;i<a.n;i++) {
        if(!fgets(line,sizeof(line),f)) die("missing dataset row");
        char *t=strtok(line,","); if(!t) die("missing label");
        a.y[i]=strtod(t,NULL); if(a.y[i]!=-1.0 && a.y[i]!=1.0) die("bad label");
        a.x[(size_t)i*a.d]=1.0;
        for(int j=1;j<a.d;j++) {
            t=strtok(NULL,","); if(!t) die("missing feature");
            double v=strtod(t,NULL); if(!isfinite(v) || v<0 || v>1) die("bad pixel");
            a.x[(size_t)i*a.d+j]=v;
        }
        if(strtok(NULL,",")) die("extra feature");
    }
    fclose(f); return a;
}
static void release(Data a) { free(a.x); free(a.y); }
static double prediction(const double *w,const double *x,int d) {
    double z=0; for(int j=0;j<d;j++) z+=w[j]*x[j]; return tanh(z);
}
static double one_loss(const double *w,const double *x,double y,int d) {
    double e=prediction(w,x,d)-y; return 0.5*e*e;
}
static void gradient(const double *w,const double *x,double y,int d,double *g) {
    double p=prediction(w,x,d), factor=(p-y)*(1-p*p);
    for(int j=0;j<d;j++) g[j]=factor*x[j];
}
static Scores evaluate(const double *w,Data a) {
    Scores s={0,0}; int correct=0;
    for(int i=0;i<a.n;i++) {
        double p=prediction(w,a.x+(size_t)i*a.d,a.d);
        double e=p-a.y[i]; s.loss+=0.5*e*e;
        if((p>=0 ? 1.0:-1.0)==a.y[i]) correct++;
    }
    s.loss/=a.n; s.accuracy=(double)correct/a.n;
    if(!isfinite(s.loss)) die("nonfinite loss");
    return s;
}
static void adam_step(double *w,double *m,double *v,const double *g,
                      int d,double lr,unsigned long long step) {
    double c1=1-pow(0.9,(double)step), c2=1-pow(0.999,(double)step);
    for(int j=0;j<d;j++) {
        m[j]=0.9*m[j]+0.1*g[j]; v[j]=0.999*v[j]+0.001*g[j]*g[j];
        w[j]-=lr*(m[j]/c1)/(sqrt(v[j]/c2)+1e-8);
    }
}
static int selftest(void) {
    double x[]={1,0.2,-0.4,0.7}, w[]={0.11,-0.21,0.31,-0.17}, g[4];
    gradient(w,x,-1,4,g); double max_error=0, eps=1e-6;
    for(int j=0;j<4;j++) {
        double old=w[j]; w[j]=old+eps; double plus=one_loss(w,x,-1,4);
        w[j]=old-eps; double minus=one_loss(w,x,-1,4); w[j]=old;
        double e=fabs(g[j]-(plus-minus)/(2*eps)); if(e>max_error) max_error=e;
    }
    double aw[]={0,0}, m[]={0,0}, v[]={0,0}, ag[]={0.25,-0.5};
    adam_step(aw,m,v,ag,2,0.01,1); double adam_error=0;
    for(int j=0;j<2;j++) {
        double expected=-0.01*ag[j]/(fabs(ag[j])+1e-8);
        double e=fabs(aw[j]-expected); if(e>adam_error) adam_error=e;
    }
    printf("{\"gradient_max_abs_error\":%.17g,\"adam_first_step_max_abs_error\":%.17g,"
           "\"passed\":%s}\n",max_error,adam_error,
           max_error<1e-7 && adam_error<1e-12 ? "true":"false");
    return max_error<1e-7 && adam_error<1e-12 ? 0:1;
}
int main(int argc,char **argv) {
    if(argc==2 && strcmp(argv[1],"--selftest")==0) return selftest();
    if(argc!=11 || strcmp(argv[1],"--train")!=0)
        die("usage: optimizer --train gd|sgd|adam lr epochs seed train.csv val.csv test.csv|- trace.csv model.bin");
    const char *method=argv[2]; int gd=strcmp(method,"gd")==0;
    int adam=strcmp(method,"adam")==0;
    if(!gd && !adam && strcmp(method,"sgd")!=0) die("unknown optimizer");
    double lr=strtod(argv[3],NULL); int epochs=atoi(argv[4]);
    if(!isfinite(lr) || lr<=0 || epochs<1 || epochs>10000) die("bad hyperparameters");
    state=(uint32_t)strtoul(argv[5],NULL,10); if(!state) state=1;
    Data train=load(argv[6]), val=load(argv[7]), test={0,0,NULL,NULL};
    int use_test=strcmp(argv[8],"-")!=0;
    if(use_test) test=load(argv[8]);
    if(train.d!=val.d || (use_test && train.d!=test.d)) die("dimension mismatch");
    int d=train.d; double *w=alloc((size_t)d), *g=alloc((size_t)d);
    double *sum=alloc((size_t)d), *m=alloc((size_t)d), *v=alloc((size_t)d);
    int *order=malloc((size_t)train.n*sizeof(int)); if(!order) die("allocation failed");
    for(int j=0;j<d;j++) w[j]=(((double)rnd()/UINT32_MAX)*2-1)*0.001;
    FILE *trace=fopen(argv[9],"w"); if(!trace) die("trace open failed");
    fprintf(trace,"epoch,updates,train_loss,train_accuracy,validation_loss,validation_accuracy,c_clock_seconds\n");
    unsigned long long updates=0; clock_t start=clock();
    Scores ts=evaluate(w,train), vs=evaluate(w,val);
    fprintf(trace,"0,0,%.17g,%.17g,%.17g,%.17g,0\n",ts.loss,ts.accuracy,vs.loss,vs.accuracy);
    for(int e=1;e<=epochs;e++) {
        for(int i=0;i<train.n;i++) order[i]=i;
        if(!gd) {
            for(int i=train.n-1;i>0;i--) {
                int j=(int)(rnd()%(uint32_t)(i+1)), tmp=order[i];
                order[i]=order[j]; order[j]=tmp;
            }
        } else memset(sum,0,(size_t)d*sizeof(double));
        for(int k=0;k<train.n;k++) {
            int i=order[k]; gradient(w,train.x+(size_t)i*d,train.y[i],d,g);
            if(gd) for(int j=0;j<d;j++) sum[j]+=g[j];
            else {
                updates++;
                if(adam) adam_step(w,m,v,g,d,lr,updates);
                else for(int j=0;j<d;j++) w[j]-=lr*g[j];
            }
        }
        if(gd) {
            updates++; for(int j=0;j<d;j++) w[j]-=lr*sum[j]/train.n;
        }
        for(int j=0;j<d;j++) if(!isfinite(w[j])) die("nonfinite weight");
        ts=evaluate(w,train); vs=evaluate(w,val);
        fprintf(trace,"%d,%llu,%.17g,%.17g,%.17g,%.17g,%.9f\n",
                e,updates,ts.loss,ts.accuracy,vs.loss,vs.accuracy,
                (double)(clock()-start)/CLOCKS_PER_SEC);
    }
    fclose(trace); FILE *model=fopen(argv[10],"wb"); if(!model) die("model open failed");
    if(fwrite(w,sizeof(double),(size_t)d,model)!=(size_t)d) die("model write failed");
    fclose(model);
    Scores te={0,0}; if(use_test) te=evaluate(w,test);
    printf("{\"optimizer\":\"%s\",\"lr\":%.17g,\"epochs\":%d,\"updates\":%llu,"
           "\"train_loss\":%.17g,\"train_accuracy\":%.17g,\"validation_loss\":%.17g,"
           "\"validation_accuracy\":%.17g,\"test_evaluated\":%s,\"test_loss\":%.17g,"
           "\"test_accuracy\":%.17g,\"c_clock_seconds\":%.9f}\n",
           method,lr,epochs,updates,ts.loss,ts.accuracy,vs.loss,vs.accuracy,
           use_test?"true":"false",te.loss,te.accuracy,(double)(clock()-start)/CLOCKS_PER_SEC);
    free(w);free(g);free(sum);free(m);free(v);free(order);
    release(train);release(val); if(use_test) release(test); return 0;
}
