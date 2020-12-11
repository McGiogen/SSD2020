import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';
import { PortafoglioComponent } from './portafoglio/portafoglio.component';
import { PrevisioneComponent } from './previsione/previsione.component';
import { RestFormComponent } from './rest-form/rest-form.component';


const routes: Routes = [
  {
    path: '',
    pathMatch: 'full',
    component: PortafoglioComponent,
  },
  {
    path: 'previsione',
    component: PrevisioneComponent,
  },
  {
    path: 'rest-form',
    component: RestFormComponent,
  },
  {
    path: '**',
    redirectTo: '',
  }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
