
import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpErrorInterceptor } from './http-error.interceptor';
import { HTTP_INTERCEPTORS } from '@angular/common/http';

@NgModule({
  declarations: [],
  imports: [ CommonModule ],
  exports: [],
  // register the classes for the error interception here
  providers: [
    {
      // interceptor for HTTP errors
      provide: HTTP_INTERCEPTORS,
      useClass: HttpErrorInterceptor,
      multi: true, // multiple interceptors are possible
    },
  ],
})
export class CoreModule {}
