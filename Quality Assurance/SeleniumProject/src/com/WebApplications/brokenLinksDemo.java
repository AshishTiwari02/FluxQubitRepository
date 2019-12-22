package com.WebApplications;

import java.net.HttpURLConnection;
import java.net.URL;
import java.util.List;
import java.util.concurrent.TimeUnit;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.chrome.ChromeDriver;

public class brokenLinksDemo {

	public static void main(String[] args) throws InterruptedException {
		
		String driverPath = "webdriver.chrome.driver";
		String driverURL = "D:\\Softwares\\Selenium\\chromedriver_win32\\chromedriver.exe";
		
		System.setProperty(driverPath, driverURL);
		WebDriver driver = new ChromeDriver();
		
		driver.manage().window().maximize();
		driver.manage().timeouts().implicitlyWait(3, TimeUnit.SECONDS);
		
		String baseUrlForLinks ="http://newtours.demoaut.com/";
		
		try {
			driver.get(baseUrlForLinks);
			Thread.sleep(3000);
			List <WebElement> allLinks = driver.findElements(By.tagName("a"));
			System.out.println("Total Link : "+ allLinks.size());
			for (int i=0;i<allLinks.size();i++)
			{
				WebElement elm = allLinks.get(i);
				String url = elm.getAttribute("href");
				//Type casting for URL links
				URL singleUrl = new URL(url);
				
				//Type cast and convert object to HTTP request
				HttpURLConnection httpCon = (HttpURLConnection) singleUrl.openConnection();
				Thread.sleep(2000);
				httpCon.connect();
				
				int responceCode = httpCon.getResponseCode();
				if (responceCode>400)
				{
					System.out.println("Link [ "+url+" ] is broken link");
				}
				else {
					System.out.println("Link [ "+url+" ] authenticate successfull. ");
				}	
			}	
		}
		
		catch(Exception e)
		{
			System.out.println(e);
		}
		finally {
			
			driver.close();
			driver.quit();
		}
		
		
		
		
		
	}
}
